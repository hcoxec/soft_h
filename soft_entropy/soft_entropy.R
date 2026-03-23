soft_entropy <- function(z, n_bins = 100, seed = 0) {
  # Soft entropy estimator from Conklin (2025).
  #
  # Estimates Shannon entropy of a distribution over embeddings by:
  # 1. Normalizing embeddings onto the unit sphere
  # 2. Sampling n random reference points from the unit sphere
  # 3. Computing cosine similarities (softly assigning each embedding to points)
  # 4. Averaging assignments over the batch to get a probability distribution
  # 5. Computing Shannon entropy of that distribution
  #
  # Args:
  #   z:      matrix of shape [batch, d]
  #   n_bins: number of reference points to sample
  #   seed:   random seed for sampling reference points
  #
  # Returns:
  #   scalar entropy estimate (efficiency-normalized, in [0, 1])

  d <- ncol(z)

  # temperature calibrated to prevent saturation across dimensionalities
  # epsilon* = 1 / sqrt(2 * d * log(n))
  temp <- 1.0 / sqrt(2 * d * log(n_bins))

  # normalize embeddings to unit sphere
  norms <- sqrt(rowSums(z^2))
  z_norm <- z / norms  # [batch, d]

  # sample reference points uniformly from unit sphere
  # (sample standard normal, then normalize)
  set.seed(seed)
  w <- matrix(rnorm(n_bins * d), nrow = n_bins, ncol = d)
  w_norms <- sqrt(rowSums(w^2))
  w <- w / w_norms  # [n_bins, d]

  # cosine similarity: [batch, n_bins]
  scores <- z_norm %*% t(w)

  # soft assignment via softmax with temperature (subtract max for stability)
  scores_scaled <- scores / temp
  scores_scaled <- scores_scaled - apply(scores_scaled, 1, max)
  exp_scores <- exp(scores_scaled)
  p_per_sample <- exp_scores / rowSums(exp_scores)  # [batch, n_bins]

  # aggregate over batch to get distribution over bins
  p <- colMeans(p_per_sample)  # [n_bins]

  # Shannon entropy
  h <- -sum(p * log(pmax(p, 1e-9)))

  # normalize by entropy of uniform distribution (efficiency)
  h_uniform <- log(n_bins)
  return(h / h_uniform)
}


soft_mutual_information <- function(z, labels, n_bins = 100, seed = 0) {
  # Estimates I(X; Z) = H(Z) - sum_x P(X=x) H(Z | X=x).
  #
  # Args:
  #   z:      matrix of shape [batch, d]
  #   labels: integer class label vector of length batch
  #   n_bins: number of reference points
  #   seed:   random seed for sampling reference points
  #
  # Returns:
  #   scalar mutual information estimate (efficiency-normalized)

  h_z <- soft_entropy(z, n_bins, seed)

  conditional_h <- 0.0
  for (label in unique(labels)) {
    mask <- labels == label
    p_label <- mean(mask)
    h_z_given_label <- soft_entropy(z[mask, , drop = FALSE], n_bins, seed)
    conditional_h <- conditional_h + p_label * h_z_given_label
  }

  return(h_z - conditional_h)
}
