# glppls_synthetic.R -- Synthetic data generation for G-LPPLS-NN
#
# Generates distributional training data following Ma & Li (2024):
#   - Labels: {mu_tc, sigma_tc, mu_A, sigma_A} (distributional targets)
#   - 10 parameter sets per label group, mixed 1st/2nd/3rd order LPPLS
#   - AR(1) + GPD noise model
#   - Jitter augmentation

library(data.table)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))
source(file.path(root, "R", "lppls_higher_order.R"))
source(file.path(root, "R", "lppls_synthetic.R"))  # for minmax_scale

# --- GPD sampling (inline inverse-CDF) ---------------------------------------

#' Sample from Generalised Pareto Distribution via inverse-CDF.
#'
#' @param n Number of samples.
#' @param xi Shape parameter.
#' @param sigma Scale parameter (> 0).
#' @return Numeric vector of GPD samples.
rgpd_inline <- function(n, xi, sigma) {
  u <- runif(n)
  if (abs(xi) < 1e-10) {
    -sigma * log(1 - u)
  } else {
    sigma / xi * ((1 - u)^(-xi) - 1)
  }
}

# --- AR(1) + GPD noise -------------------------------------------------------

#' Add AR(1) + GPD noise to a time series.
#'
#' Noise model: eta_t = phi * eta_{t-1} + sign_t * eps_t + eps'_t
#' where eps_t ~ GPD and eps'_t ~ N(0, delta^2).
#'
#' @param values Numeric vector of clean series values.
#' @param phi AR(1) coefficient.
#' @param delta Gaussian noise standard deviation.
#' @param xi GPD shape parameter.
#' @param sigma_gpd GPD scale parameter.
#' @return Numeric vector of noisy values.
add_ar1_gpd_noise <- function(values, phi, delta, xi, sigma_gpd) {
  n <- length(values)
  eta <- numeric(n)
  gpd_vals <- rgpd_inline(n, xi, sigma_gpd)
  gauss_vals <- rnorm(n, 0, delta)
  signs <- sample(c(-1, 1), n, replace = TRUE)

  eta[1] <- signs[1] * gpd_vals[1] + gauss_vals[1]
  for (i in 2:n) {
    eta[i] <- phi * eta[i - 1] + signs[i] * gpd_vals[i] + gauss_vals[i]
  }
  values + eta
}

# --- Label generation ---------------------------------------------------------

#' Generate distributional labels for G-LPPLS-NN training.
#'
#' Each label has {mu_tc, sigma_tc, mu_A, sigma_A} following ranges in
#' Appendix B.1 of Ma & Li (2024).
#'
#' @param n Number of label groups.
#' @param t_len Length of time series.
#' @return data.table with columns mu_tc, sigma_tc, mu_A, sigma_A.
generate_glppls_labels <- function(n, t_len = 252L) {
  mu_tc <- runif(n, t_len, t_len + 50)          # U(252, 302)
  sigma_tc <- runif(n, 1, 14)
  exp_mu_A <- runif(n, exp(1), 120000)
  mu_A <- log(exp_mu_A)
  exp_sigma_A <- runif(n, 0, 0.025 * exp_mu_A)
  sigma_A <- log(pmax(exp_sigma_A, 1e-10))

  data.table(mu_tc = mu_tc, sigma_tc = sigma_tc,
             mu_A = mu_A, sigma_A = sigma_A)
}

# --- Parameter generation from labels ----------------------------------------

#' Generate LPPLS parameter sets from a single distributional label.
#'
#' Draws n_sets of {tc, A} from the label's distributions, plus random
#' nonlinear parameters appropriate for the given LPPLS order.
#'
#' @param label_row Single-row data.table with mu_tc, sigma_tc, mu_A, sigma_A.
#' @param n_sets Number of parameter sets per label (default 10).
#' @param order LPPLS order: 1, 2, or 3.
#' @param t_len Time series length.
#' @return data.table of parameter sets.
generate_params_from_label <- function(label_row, n_sets = 10L, order = 1L,
                                       t_len = 252L) {
  tc_vals <- rnorm(n_sets, label_row$mu_tc, label_row$sigma_tc)
  tc_vals <- pmax(tc_vals, t_len + 1)  # ensure tc > t_len

  exp_A_vals <- rnorm(n_sets, exp(label_row$mu_A),
                      exp(label_row$sigma_A))
  exp_A_vals <- pmax(exp_A_vals, 1)
  A_vals <- log(exp_A_vals)

  # Shared nonlinear params (Appendix B.1)
  alpha_vals <- runif(n_sets, 0.01, 0.99)
  omega_vals <- runif(n_sets, 4.9, 15)
  B_vals <- sample(c(-1, 1), n_sets, replace = TRUE) *
    runif(n_sets, 0.01, 1)
  C1_vals <- runif(n_sets, -1, 1)
  C2_vals <- runif(n_sets, -1, 1)

  dt <- data.table(
    tc = tc_vals, alpha = alpha_vals, omega = omega_vals,
    A = A_vals, B = B_vals, C1 = C1_vals, C2 = C2_vals,
    order = order
  )

  if (order >= 2L) {
    dt_range <- t_len
    dt[, delta_t := runif(n_sets, 0.2 * dt_range, 1.2 * dt_range)]
    dt[, delta_omega := runif(n_sets, -75, 6)]
  }

  if (order >= 3L) {
    dt_range <- t_len
    dt[, delta_t_prime := runif(n_sets, 0.5 * dt_range, 1.5 * dt_range)]
    dt[, delta_omega_prime := runif(n_sets, -100, 4.9)]
  }

  dt
}

# --- Series generation for any order -----------------------------------------

generate_glppls_series <- function(params, t_len = 252L) {
  t_seq <- seq(0, t_len - 1)
  ord <- params$order

  if (ord == 1L) {
    vals <- lppls_value(t_seq, params$tc, params$alpha, params$omega,
                        params$A, params$B, params$C1, params$C2)
  } else if (ord == 2L) {
    vals <- lppls_value_order2(t_seq, params$tc, params$alpha, params$omega,
                               params$delta_t, params$delta_omega,
                               params$A, params$B, params$C1, params$C2)
  } else {
    vals <- lppls_value_order3(t_seq, params$tc, params$alpha, params$omega,
                               params$delta_t, params$delta_omega,
                               params$delta_t_prime, params$delta_omega_prime,
                               params$A, params$B, params$C1, params$C2)
  }
  vals
}

# --- Jitter augmentation -----------------------------------------------------

#' Apply jitter augmentation by shifting and interpolating.
#'
#' Shifts series forward by up to 5% or backward by up to 20%, then
#' linearly interpolates back to original length.
#'
#' @param series Numeric vector.
#' @param t_len Target output length.
#' @return Numeric vector of length t_len.
apply_jitter <- function(series, t_len = 252L) {
  n <- length(series)
  shift_frac <- runif(1, -0.05, 0.20)  # forward 5% to backward 20%
  shift <- round(shift_frac * n)

  if (shift > 0) {
    # Backward shift: truncate end
    truncated <- series[seq_len(n - shift)]
  } else if (shift < 0) {
    # Forward shift: truncate beginning
    truncated <- series[seq(abs(shift) + 1, n)]
  } else {
    truncated <- series
  }

  if (length(truncated) < 2) return(rep(series[1], t_len))

  # Interpolate back to t_len
  approx(seq_along(truncated), truncated, n = t_len)$y
}

# --- Full dataset generation --------------------------------------------------

#' Generate G-LPPLS-NN training dataset.
#'
#' @param n_labels Number of distributional label groups.
#' @param t_len Time series length.
#' @param n_sets Number of parameter sets per label group.
#' @param order_mix Integer vector of relative weights for 1st:2nd:3rd order.
#' @param phi AR(1) coefficient for noise.
#' @param delta Gaussian noise sd.
#' @param xi GPD shape parameter.
#' @param sigma_gpd GPD scale parameter.
#' @param seed Random seed.
#' @return list(X, Y, labels) where X is (n_total, t_len), Y is (n_total, 4).
generate_glppls_dataset <- function(n_labels = 12000L,
                                    t_len = 252L,
                                    n_sets = 10L,
                                    order_mix = c(2, 1, 1),
                                    phi = 0.9,
                                    delta = 0.01,
                                    xi = 0.5,
                                    sigma_gpd = 0.01,
                                    seed = 42) {
  set.seed(seed)
  labels_dt <- generate_glppls_labels(n_labels, t_len)

  # Assign orders based on mix ratio
  order_probs <- order_mix / sum(order_mix)
  orders <- sample(1:3, n_labels, replace = TRUE, prob = order_probs)

  n_total <- n_labels * n_sets
  X <- matrix(0, nrow = n_total, ncol = t_len)
  Y <- matrix(0, nrow = n_total, ncol = 4)

  row_idx <- 1L
  for (i in seq_len(n_labels)) {
    label <- labels_dt[i]
    params_dt <- generate_params_from_label(label, n_sets, orders[i], t_len)

    for (j in seq_len(n_sets)) {
      p <- params_dt[j]
      vals <- tryCatch(
        generate_glppls_series(p, t_len),
        error = function(e) rep(NA_real_, t_len)
      )

      if (any(!is.finite(vals))) {
        vals <- rep(0, t_len)
      }

      # Add AR(1) + GPD noise
      noisy <- add_ar1_gpd_noise(vals, phi, delta * sd(vals + 1e-10),
                                 xi, sigma_gpd * sd(vals + 1e-10))

      # Jitter augmentation
      noisy <- apply_jitter(noisy, t_len)

      # Min-max scale input
      X[row_idx, ] <- minmax_scale(noisy)

      # Normalized labels: mu_tc/sigma_tc by t_len, mu_A/sigma_A by end value
      end_val <- abs(noisy[t_len]) + 1e-10
      Y[row_idx, ] <- c(
        label$mu_tc / t_len,
        label$sigma_tc / t_len,
        label$mu_A / log(end_val + 1),
        exp(label$sigma_A) / (end_val + 1e-10)
      )

      row_idx <- row_idx + 1L
    }
  }

  list(X = X, Y = Y, labels = labels_dt)
}

#' Generate G-LPPLS-NN training dataset in parallel.
#'
#' @inheritParams generate_glppls_dataset
#' @return list(X, Y, labels).
generate_glppls_dataset_parallel <- function(n_labels = 12000L,
                                             t_len = 252L,
                                             n_sets = 10L,
                                             order_mix = c(2, 1, 1),
                                             phi = 0.9,
                                             delta = 0.01,
                                             xi = 0.5,
                                             sigma_gpd = 0.01,
                                             seed = 42) {
  set.seed(seed)
  labels_dt <- generate_glppls_labels(n_labels, t_len)
  order_probs <- order_mix / sum(order_mix)
  orders <- sample(1:3, n_labels, replace = TRUE, prob = order_probs)

  results <- future.apply::future_lapply(seq_len(n_labels), function(i) {
    label <- labels_dt[i]
    params_dt <- generate_params_from_label(label, n_sets, orders[i], t_len)

    X_block <- matrix(0, nrow = n_sets, ncol = t_len)
    Y_block <- matrix(0, nrow = n_sets, ncol = 4)

    for (j in seq_len(n_sets)) {
      p <- params_dt[j]
      vals <- tryCatch(
        generate_glppls_series(p, t_len),
        error = function(e) rep(NA_real_, t_len)
      )

      if (any(!is.finite(vals))) vals <- rep(0, t_len)

      noisy <- add_ar1_gpd_noise(vals, phi, delta * sd(vals + 1e-10),
                                 xi, sigma_gpd * sd(vals + 1e-10))
      noisy <- apply_jitter(noisy, t_len)
      X_block[j, ] <- minmax_scale(noisy)

      end_val <- abs(noisy[t_len]) + 1e-10
      Y_block[j, ] <- c(
        label$mu_tc / t_len,
        label$sigma_tc / t_len,
        label$mu_A / log(end_val + 1),
        exp(label$sigma_A) / (end_val + 1e-10)
      )
    }

    list(X = X_block, Y = Y_block)
  }, future.seed = TRUE)

  X <- do.call(rbind, lapply(results, `[[`, "X"))
  Y <- do.call(rbind, lapply(results, `[[`, "Y"))

  list(X = X, Y = Y, labels = labels_dt)
}
