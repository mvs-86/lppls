# lppls_synthetic.R -- Synthetic LPPLS data generation for training/evaluation
#
# Generates noisy LPPLS time series with known parameters for:
# - P-LNN training (white noise, AR1 noise, both)
# - Model evaluation against known ground truth

library(data.table)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

# --- Parameter generation ---------------------------------------------------

generate_lppls_params <- function(n,
                                  t_len = 252,
                                  tc_range = NULL,
                                  m_range = c(0.1, 0.9),
                                  omega_range = c(6, 13)) {
  if (is.null(tc_range)) tc_range <- c(t_len, t_len + 50)
  data.table(
    tc    = runif(n, tc_range[1], tc_range[2]),
    m     = runif(n, m_range[1], m_range[2]),
    omega = runif(n, omega_range[1], omega_range[2]),
    A     = runif(n, 5, 10),
    B     = runif(n, -1, -0.01),
    C1    = runif(n, -0.05, 0.05),
    C2    = runif(n, -0.05, 0.05)
  )
}

# --- Series generation ------------------------------------------------------

generate_lppls_series <- function(params, t_len = 252) {
  t_seq <- seq(0, t_len - 1)
  vals <- lppls_value(t_seq, params$tc, params$m, params$omega,
                      params$A, params$B, params$C1, params$C2)
  data.table(t = t_seq, value = vals)
}

# --- Noise functions --------------------------------------------------------

add_white_noise <- function(values, amplitude) {
  sigma <- amplitude * sd(values)
  values + rnorm(length(values), 0, sigma)
}

add_ar1_noise <- function(values, amplitude, phi = 0.9) {
  n <- length(values)
  sigma_target <- amplitude * sd(values)
  sigma_eps <- sigma_target * sqrt(1 - phi^2)
  eta <- numeric(n)
  eta[1] <- rnorm(1, 0, sigma_eps)
  for (i in 2:n) {
    eta[i] <- phi * eta[i - 1] + rnorm(1, 0, sigma_eps)
  }
  values + eta
}

# --- Min-max scaling --------------------------------------------------------

minmax_scale <- function(x) {
  rng <- range(x)
  if (rng[2] == rng[1]) return(rep(0.5, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}

# --- Dataset generation -----------------------------------------------------

generate_training_dataset <- function(n = 10000,
                                      t_len = 252,
                                      noise_type = c("white", "ar1", "both"),
                                      noise_amp_white = c(0.01, 0.15),
                                      noise_amp_ar1 = c(0.01, 0.05),
                                      phi = 0.9,
                                      seed = 42) {
  noise_type <- match.arg(noise_type)
  set.seed(seed)

  params_dt <- generate_lppls_params(n, t_len)

  # Normalize targets: tc to [0,1], m and omega already bounded
  tc_max <- t_len + 50
  Y <- as.matrix(params_dt[, .(
    tc_norm = tc / tc_max,
    m       = m,
    omega   = omega / 13  # normalize to [0,1]-ish range
  )])

  X <- matrix(0, nrow = n, ncol = t_len)

  for (i in seq_len(n)) {
    p <- params_dt[i]
    series <- generate_lppls_series(p, t_len)
    vals <- series$value

    amp_w <- runif(1, noise_amp_white[1], noise_amp_white[2])
    amp_a <- runif(1, noise_amp_ar1[1], noise_amp_ar1[2])

    noisy <- switch(noise_type,
      white = add_white_noise(vals, amp_w),
      ar1   = add_ar1_noise(vals, amp_a, phi),
      both  = {
        if (runif(1) < 0.5) {
          add_white_noise(vals, amp_w)
        } else {
          add_ar1_noise(vals, amp_a, phi)
        }
      }
    )
    X[i, ] <- minmax_scale(noisy)
  }

  list(X = X, Y = Y, params = params_dt)
}

generate_training_dataset_parallel <- function(n = 10000,
                                               t_len = 252,
                                               noise_type = c("white", "ar1", "both"),
                                               noise_amp_white = c(0.01, 0.15),
                                               noise_amp_ar1 = c(0.01, 0.05),
                                               phi = 0.9,
                                               seed = 42) {
  noise_type <- match.arg(noise_type)
  set.seed(seed)

  params_dt <- generate_lppls_params(n, t_len)
  tc_max <- t_len + 50

  Y <- as.matrix(params_dt[, .(
    tc_norm = tc / tc_max,
    m       = m,
    omega   = omega / 13
  )])

  # Pre-generate noise amplitudes
  amps_w <- runif(n, noise_amp_white[1], noise_amp_white[2])
  amps_a <- runif(n, noise_amp_ar1[1], noise_amp_ar1[2])
  coin <- runif(n)

  results <- future.apply::future_lapply(seq_len(n), function(i) {
    p <- params_dt[i]
    series <- generate_lppls_series(p, t_len)
    vals <- series$value

    noisy <- switch(noise_type,
      white = add_white_noise(vals, amps_w[i]),
      ar1   = add_ar1_noise(vals, amps_a[i], phi),
      both  = {
        if (coin[i] < 0.5) add_white_noise(vals, amps_w[i])
        else add_ar1_noise(vals, amps_a[i], phi)
      }
    )
    minmax_scale(noisy)
  }, future.seed = TRUE)

  X <- do.call(rbind, results)
  list(X = X, Y = Y, params = params_dt)
}
