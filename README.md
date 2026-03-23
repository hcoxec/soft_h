# Soft Entropy Estimator

[![PyPI version](https://img.shields.io/pypi/v/soft-entropy)](https://pypi.org/project/soft-entropy/)
[![PyPI downloads](https://img.shields.io/pypi/dm/soft-entropy)](https://pypi.org/project/soft-entropy/)

Implementation of the soft entropy estimator from [Conklin (2025)](https://arxiv.org/pdf/2505.23960), as used in *Learning is Forgetting: LLM Training as Lossy Compression* (Conklin et al., ICLR 2026).

---

## Installation

```bash
pip install soft-entropy
```

JAX and PyTorch are included by default. To also install Jupyter:

```bash
pip install soft-entropy[nb]
```

On Google Colab, where Jupyter is pre-installed, the default install works as-is.

To install from source:

```bash
git clone https://github.com/...
pip install -e .
```

You can run a tutorial estimating entropies on MNIST locally using the notebook in this repo, or on google collab [here](https://colab.research.google.com/drive/1js6TuiT5IeMWjDxIIiZo3yXOwgstR7uO?usp=sharing)

---

## How it works

A visual depiction of this process can be found [here](https://blog.hconklin.com/posts/so-u-want/)

Soft Entropy estimates the entropy of a distribution over embeddings without storing all embeddings in memory. The procedure for a set of embeddings `Z` of dimensionality `d`:

1. **Normalise** each embedding onto the unit sphere.
2. **Sample** `n` reference points `W` uniformly from the unit sphere (once, by drawing from a standard normal and normalising).
3. **Compute Distances** comparing each `z` to all the reference points `W`. Usually cosine similarites are used.
4. **Soft-assign** `z` to reference points by passing the distances through a softmax with a calibrated temperature `ε* = 1 / sqrt(2d · log(n))`. This gives each embedding a probability vector over the `n` bins.
5. **Aggregate & Compute Entropy** we average the probability vectors over the batch to get a single distribution `P(Z)` over bins. Computing the Shannon entropy `H(Z)` is then just `-Σ P(Z) log P(Z)`. In practice we usually normalise this by maximum entropy `log(n)` so it lives in `[0, 1]` (called *efficiency*).

Conditioning on a label `x` is done by repeating step 5 on only the embeddings where the label equals `x`. Mutual information then follows as:

```
I(X; Z) = H(Z) - Σ_x P(X=x) · H(Z | X=x)
```

The temperature calibration ensures estimates are directly comparable across models with different hidden dimensionalities.

---

## Package structure

```
soft_entropy/
├── pytorch.py      # PyTorch implementation
├── numpy.py        # NumPy implementation
├── jax.py          # JAX implementation
└── accumulator.py  # Backend-agnostic batch accumulator

soft_entropy.R      # R implementation
examples_MNIST.ipynb
```

---

## Single-batch functions

Each backend module exposes two functions with the same interface:

```python
soft_entropy(z, n_bins=100, seed=0) -> scalar
soft_mutual_information(z, labels, n_bins=100, seed=0) -> scalar
```

- `z`: embeddings `[batch, d]`
- `labels`: 1-D integer array of class labels `[batch]`
- `n_bins`: number of reference points (default 100; robust to this choice)
- `seed`: random seed used to sample the reference points `W`

**The seed is the key to cross-model comparability.** Because entropy is measured relative to the same fixed reference points, two calls with the same `seed`, `d` and `n_bins` are operating in the same "coordinate system" — so their H(Z) and I(X;Z) values are directly comparable even if the embeddings come from different models, layers, or datasets. Always use the same seed when comparing estimates.

These are appropriate when all embeddings fit in memory at once.

**PyTorch:**
```python
from soft_entropy.pytorch import soft_entropy, soft_mutual_information

h  = soft_entropy(z)
mi = soft_mutual_information(z, labels)
```

**NumPy:**
```python
from soft_entropy.numpy import soft_entropy, soft_mutual_information

h  = soft_entropy(z_np)
mi = soft_mutual_information(z_np, labels_np)
```

**JAX:**
```python
from soft_entropy.jax import soft_entropy, soft_mutual_information

h  = soft_entropy(z_jax)
mi = soft_mutual_information(z_jax, labels_jax)
```

---

## Accumulator — large datasets and multiple label sets

`SoftEntropyAccumulator` accumulates soft assignment counts across batches. The reference points `W` are sampled once at construction and reused every batch, so counts are consistent across the whole dataset.

When using the `torch` backend, the accumulator automatically selects the best available device (CUDA → MPS → CPU) and moves all tensors and incoming batches there. No manual device management is required.

```python
from soft_entropy import SoftEntropyAccumulator

acc = SoftEntropyAccumulator(d=768, n_bins=100, seed=0, backend='torch')
```

**`backend`** is one of `'torch'`, `'numpy'`, or `'jax'`. The input arrays passed to `update()` must match.

### Basic usage — single label set

```python
for z, labels in dataloader:
    acc.update(z, labels)

print(acc.entropy())             # float — H(Z)
print(acc.mutual_information())  # dict — {"labels": float}
print(acc.results())             # flat summary dict
```

### Multiple label sets in one pass

Pass a dict to `labels`. The soft assignments are computed once per batch; accumulating extra label sets costs only cheap indexing.

```python
for z, token_ids, bigram_ids, pref_labels in dataloader:
    acc.update(z, labels={
        "token":      token_ids,
        "bigram":     bigram_ids,
        "preference": pref_labels,
    })

acc.entropy()
# 0.863

acc.mutual_information()
# {"token": 0.12, "bigram": 0.31, "preference": 0.04}

acc.conditional_entropy()
# {"token":      {0: 0.71, 1: 0.69, ...},
#  "bigram":     {(0,1): 0.58, ...},
#  "preference": {0: 0.82, 1: 0.79}}

acc.results()
# {"H(Z)": 0.863,
#  "I(X;Z)/token": 0.12,      "regularity/token": 0.14,
#  "I(X;Z)/bigram": 0.31,     "regularity/bigram": 0.36,
#  "I(X;Z)/preference": 0.04, "regularity/preference": 0.05}
```

### Label structure

Labels are always **flat 1-D integer arrays** of length `batch`. What constitutes a label is up to the caller:

| Use case | Label value |
|---|---|
| Token-level back-off | Token id at each position |
| Bigram back-off | Integer encoding of `(prev_token, token)` pair |
| Preference | Binary `0` (rejected) / `1` (preferred) |
| Language id | Integer language code |
| Digit class | Class integer `0–9` |

For n-gram back-off as used in the paper, encode each n-gram as a single label (e.g.  `(current_token,prev_token)` for bigrams) and pass one label set per n-gram width.

### Resetting between runs

```python
acc.reset()   # clears all counts, keeps reference points W
```

---

## Efficiency, Regularity, and Optimality

*Efficiency* is entropy normalised by its maximum possible value:

```
Efficiency = H(Z) / log(n)
```

All entropy and mutual information values returned by this library are already efficiency-normalised (i.e. in `[0, 1]`), so they are directly comparable across models with different embedding dimensionalities or numbers of reference points.

*Regularity* is mutual information normalised by entropy:

```
Regularity = I(X; Z) / H(Z)
```

`acc.results()` returns regularity for each label set under the key `regularity/<label_set>`.

*Optimality* is the ratio of expressivity to complexity:

```
Optimality = I(Y; Z) / I(X; Z)
```

where `X` is the input label (e.g. token) and `Y` is the output label (e.g. next token). This approaches 1.0 as representations converge to the Information Bottleneck bound. Computing optimality requires two separate label sets (input and output) and dividing their mutual information estimates.

---

## Notebooks

| Notebook | Contents |
|---|---|
| `examples_MNIST.ipynb` | Examples on sklearn digits dataset verifying estimator behaviour |
