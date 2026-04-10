"""
LLM entropy analysis via soft entropy accumulation.

Loads a causal LM and a text dataset from HuggingFace, runs batched inference,
and uses SoftEntropyAccumulator to estimate per-layer H(Z) and I(X; Z) for
unigram, bigram, and/or trigram token labels — without storing representations
in memory.

Usage::

    inferrer = LLMInferrer(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        dataset_id="sentence-transformers/reddit",
        label_types=["unigram", "bigram", "trigram"],
    )
    results = inferrer.run(n_examples=500, batch_size=4)
    print(results["mean"])        # averages across transformer layers
    print(results["per_layer"])   # list of result dicts, one per layer
"""

from __future__ import annotations
import gc
import torch
from tqdm.auto import tqdm

from soft_entropy.accumulator import SoftEntropyAccumulator


_TEXT_COLUMN_CANDIDATES = ["text", "body", "sentence", "content", "document", "passage"]

_ORDER = {"unigram": 1, "bigram": 2, "trigram": 3}


def _infer_text_column(dataset) -> str:
    """Return the name of the text column, or raise if it can't be inferred."""
    from datasets import Value
    features = dataset.features
    for name in _TEXT_COLUMN_CANDIDATES:
        if name in features:
            return name
    for name, feat in features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            return name
    raise ValueError(
        f"Cannot infer text column from features: {list(features.keys())}. "
        "Pass text_column= explicitly."
    )


def _encode_ngram(ids: torch.Tensor, start: int, end: int, order: int, V: int, forward: bool) -> torch.Tensor:
    """
    Encode n-gram labels for valid positions start..end (exclusive).

    For each valid position i:
      - forward=False (input):  n-gram ending at i   → ids[i-n+1], ..., ids[i]
      - forward=True  (output): n-gram starting at i+1 → ids[i+1], ..., ids[i+n]

    Encodes as a single int64: t0*V^(n-1) + t1*V^(n-2) + ... + t_{n-1}.
    All slices are guaranteed valid by the caller's choice of start/end.
    """
    ids = ids.long()
    if forward:
        # tokens ids[start+1:end+1], ids[start+2:end+2], ..., ids[start+order:end+order]
        code = ids[start + 1: end + 1]
        for k in range(2, order + 1):
            code = code * V + ids[start + k: end + k]
    else:
        # tokens ids[start-order+1:end-order+1], ..., ids[start:end]
        code = ids[start - order + 1: end - order + 1]
        for k in range(order - 2, -1, -1):
            offset = start - k
            code = code * V + ids[offset: end - k]
    return code


class LLMInferrer:
    """
    Estimates per-layer soft entropy and mutual information for a causal LM.

    For each layer one SoftEntropyAccumulator tracks H(Z) and I(X; Z) for
    each requested label type. Supported label types:

      - "unigram"  — current/next token id
      - "bigram"   — 2-token n-gram encoded as t0*V + t1
      - "trigram"  — 3-token n-gram encoded as t0*V^2 + t1*V + t2

    Each label type produces both an input label (what tokens produced Z_i)
    and an output label (what tokens Z_i predicts). When multiple orders are
    requested the valid position range is restricted to positions where all
    n-grams are well-defined — i.e. positions max_order-1 .. seq_len-max_order.

    Args:
        model_id:      HuggingFace model identifier
        dataset_id:    HuggingFace dataset identifier
        label_types:   list of n-gram orders to estimate; any of
                       "unigram", "bigram", "trigram"
        n_bins:        number of soft reference bins for the accumulator
        seed:          random seed for reproducible reference points
        max_length:    tokenizer truncation length
        text_column:   dataset column containing text; auto-inferred if None
        dataset_split: dataset split to use (default "train")
    """

    def __init__(
        self,
        model_id: str,
        dataset_id: str,
        label_types: list[str] = ("unigram",),
        n_bins: int = 100,
        seed: int = 0,
        max_length: int = 128,
        text_column: str | None = None,
        dataset_split: str = "train",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_id = model_id
        self.dataset_id = dataset_id
        self.label_types = list(label_types)
        self.max_length = max_length
        self.text_column = text_column
        self.dataset_split = dataset_split
        self.n_bins = n_bins
        self.seed = seed

        # highest n-gram order requested — determines the valid position range
        self.max_order = max(_ORDER[lt] for lt in self.label_types)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_hidden_states=True,
            device_map="auto",
        )
        self.model.eval()

        d = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        model_dtype = next(self.model.parameters()).dtype
        self.accs = [
            SoftEntropyAccumulator(d=d, n_bins=n_bins, seed=seed, backend="torch", dtype=model_dtype)
            for _ in range(self.n_layers + 1)  # index 0 = embedding layer
        ]

    def run(self, n_examples: int = 500, batch_size: int = 4) -> dict:
        """
        Stream the dataset, accumulate representations, and return results.

        Returns:
            {
                "per_layer": list of result dicts (index 0 = embedding layer),
                "mean":      dict of values averaged across transformer layers only
            }
        """
        from datasets import load_dataset

        dataset = load_dataset(self.dataset_id, split=self.dataset_split, streaming=True)

        if self.text_column is None:
            self.text_column = _infer_text_column(dataset)

        batch, n_processed = [], 0
        for example in tqdm(dataset, total=n_examples, desc=f"{self.model_id}"):
            text = example[self.text_column]
            if not isinstance(text, str) or not text.strip():
                continue
            batch.append(text)
            if len(batch) == batch_size:
                self._process_batch(batch)
                n_processed += len(batch)
                batch = []
                if n_processed >= n_examples:
                    break

        if batch:
            self._process_batch(batch)

        return self._collect_results()

    def _process_batch(self, texts: list[str]) -> None:
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        with torch.no_grad():
            output = self.model(**tokenized)

        hidden_states  = output.hidden_states   # tuple: (n_layers+1) x [batch, seq, d]
        input_ids      = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        n   = self.max_order
        V   = self.tokenizer.vocab_size
        # valid absolute positions for sequence of length L:
        #   start = n-1  (need n-1 prior tokens for input n-gram)
        #   end   = L-n  (exclusive; need n future tokens for output n-gram)
        label_lists: dict[str, list[torch.Tensor]] = {k: [] for k in self._label_keys()}
        seq_slices: list[tuple[int, int, int]] = []  # (seq_i, start, end)

        for i in range(len(texts)):
            seq_len = int(attention_mask[i].sum().item())
            start, end = n - 1, seq_len - n   # end is exclusive
            if end <= start:
                continue

            ids = input_ids[i, :seq_len]
            seq_slices.append((i, start, end))

            for lt in self.label_types:
                order = _ORDER[lt]
                label_lists[f"input_{lt}"].append(
                    _encode_ngram(ids, start, end, order, V, forward=False)
                )
                label_lists[f"output_{lt}"].append(
                    _encode_ngram(ids, start, end, order, V, forward=True)
                )

        if not seq_slices:
            return

        flat_labels = {k: torch.cat(v) for k, v in label_lists.items()}

        for layer_idx, acc in enumerate(self.accs):
            z_parts = [
                hidden_states[layer_idx][seq_i, start:end, :]
                for seq_i, start, end in seq_slices
            ]
            z = torch.cat(z_parts, dim=0)  # [total_positions, d]
            acc.update(z, labels=flat_labels)

        del output, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _label_keys(self) -> list[str]:
        return [f"{direction}_{lt}" for lt in self.label_types for direction in ("input", "output")]

    def _collect_results(self) -> dict:
        per_layer = [acc.results() for acc in self.accs]

        transformer_layers = per_layer[1:]  # skip embedding at index 0
        keys = transformer_layers[0].keys()
        mean = {
            k: sum(layer[k] for layer in transformer_layers) / len(transformer_layers)
            for k in keys
        }

        return {"per_layer": per_layer, "mean": mean}
