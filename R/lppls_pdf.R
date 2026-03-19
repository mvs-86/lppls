# lppls_pdf.R -- tc PDF/CDF computation across calibration windows
#
# Runs LPPLS calibrations over sliding (t1, t2) windows and computes
# kernel density estimates (PDFs) of predicted tc values.

library(data.table)
library(future.apply)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

# --- Multi-window calibration -----------------------------------------------

lppls_multi_window <- function(obs_dt,
                               t1_seq,
                               t2_seq,
                               method = c("lm", "mlnn", "plnn"),
                               model = NULL,
                               n_starts = 25,
                               min_window = 50,
                               m_range = c(0.1, 0.9),
                               omega_range = c(6, 13)) {
  method <- match.arg(method)

  # Build window grid
  grid <- CJ(t1 = t1_seq, t2 = t2_seq)
  grid <- grid[t2 - t1 >= min_window]

  results <- future_lapply(seq_len(nrow(grid)), function(i) {
    row <- grid[i]
    sub <- obs_dt[t >= row$t1 & t <= row$t2]
    if (nrow(sub) < min_window) return(NULL)

    t_vec <- sub$t
    obs_vec <- sub$value

    fit <- NULL

    if (method == "lm") {
      t2_val <- max(t_vec)
      tc_range <- c(t2_val + 1, t2_val + 0.2 * (t2_val - min(t_vec)))
      fit <- tryCatch(
        lppls_fit_lm(t_vec, obs_vec, tc_range, m_range, omega_range, n_starts),
        error = function(e) NULL
      )
    } else if (method == "plnn" && !is.null(model)) {
      # Resample to 252 points if needed
      if (length(obs_vec) != 252) {
        idx <- round(seq(1, length(obs_vec), length.out = 252))
        obs_resampled <- obs_vec[idx]
      } else {
        obs_resampled <- obs_vec
      }
      pred <- plnn_predict(model, obs_resampled)
      fit <- list(
        tc = pred$tc, m = pred$m, omega = pred$omega,
        sse = NA_real_, converged = TRUE
      )
    } else if (method == "mlnn") {
      fit <- tryCatch(
        mlnn_train(t_vec, obs_vec, epochs = 500, verbose = FALSE),
        error = function(e) NULL
      )
    }

    if (is.null(fit)) return(NULL)
    if (!lppls_filter(fit, m_range, omega_range)) return(NULL)

    data.table(
      t1 = row$t1, t2 = row$t2,
      tc = fit$tc, m = fit$m, omega = fit$omega,
      sse = fit$sse, method = method
    )
  }, future.seed = TRUE)

  rbindlist(results[!sapply(results, is.null)])
}

# --- PDF computation --------------------------------------------------------

lppls_tc_pdf <- function(tc_values, bandwidth = "SJ", n_points = 512) {
  if (length(tc_values) < 2) {
    return(data.table(tc = tc_values, density = 1))
  }
  bw <- tryCatch(bandwidth, error = function(e) "nrd0")
  d <- density(tc_values, bw = bw, n = n_points)
  data.table(tc = d$x, density = d$y)
}

# --- CDF computation -------------------------------------------------------

lppls_tc_cdf <- function(tc_values) {
  if (length(tc_values) < 1) {
    return(data.table(tc = numeric(0), cdf = numeric(0)))
  }
  tc_sorted <- sort(tc_values)
  n <- length(tc_sorted)
  data.table(tc = tc_sorted, cdf = seq_len(n) / n)
}
