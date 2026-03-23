import jax
import jax.numpy as jnp
import math


def soft_entropy(z: jax.Array, n_bins: int = 100, seed: int = 0) -> jax.Array:
    """
    Soft entropy estimator from Conklin (2025).

    Estimates Shannon entropy of a distribution over embeddings by:
    1. Normalizing embeddings onto the unit sphere
    2. Sampling n random reference points from the unit sphere
    3. Computing cosine similarities (softly assigning each embedding to points)
    4. Averaging assignments over the batch to get a probability distribution
    5. Computing Shannon entropy of that distribution

    Args:
        z: embeddings of shape [batch, d]
        n_bins: number of reference points to sample
        seed: random seed for sampling reference points

    Returns:
        scalar entropy estimate (efficiency-normalized, in [0, 1])
    """
    d = z.shape[-1]

    # temperature calibrated to prevent saturation across dimensionalities
    # epsilon* = 1 / sqrt(2 * d * log(n))
    temp = 1.0 / math.sqrt(2 * d * math.log(n_bins))

    # normalize embeddings to unit sphere
    z_norm = z / jnp.linalg.norm(z, axis=-1, keepdims=True)  # [batch, d]

    # sample reference points uniformly from unit sphere
    # (sample standard normal, then normalize)
    key = jax.random.PRNGKey(seed)
    w = jax.random.normal(key, shape=(n_bins, d))
    w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)  # [n_bins, d]

    # cosine similarity: [batch, n_bins]
    scores = z_norm @ w.T

    # soft assignment via softmax with temperature
    p_per_sample = jax.nn.softmax(scores / temp, axis=-1)  # [batch, n_bins]

    # aggregate over batch to get distribution over bins
    p = p_per_sample.mean(axis=0)  # [n_bins]

    # Shannon entropy
    h = -(p * jnp.log(jnp.clip(p, a_min=1e-9))).sum()

    # normalize by entropy of uniform distribution (efficiency)
    h_uniform = math.log(n_bins)
    return h / h_uniform


def soft_mutual_information(z: jax.Array, labels: jax.Array, n_bins: int = 100, seed: int = 0) -> jax.Array:
    """
    Estimates I(X; Z) = H(Z) - sum_x P(X=x) H(Z | X=x).

    Args:
        z: embeddings of shape [batch, d]
        labels: integer class labels of shape [batch]
        n_bins: number of reference points
        seed: random seed for sampling reference points

    Returns:
        scalar mutual information estimate (efficiency-normalized)
    """
    h_z = soft_entropy(z, n_bins, seed)

    unique_labels = jnp.unique(labels)
    conditional_h = 0.0
    for label in unique_labels:
        mask = labels == label
        p_label = mask.mean()
        h_z_given_label = soft_entropy(z[mask], n_bins, seed)
        conditional_h = conditional_h + p_label * h_z_given_label

    return h_z - conditional_h
