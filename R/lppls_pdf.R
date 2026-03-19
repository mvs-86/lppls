# lppls_pdf.R -- tc PDF/CDF computation across calibration windows
#
# Runs LPPLS calibrations over sliding (t1, t2) windows and computes
# kernel density estimates (PDFs) of predicted tc values.

library(data.table)
library(future.apply)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

# --- Multi-window calibration -----------------------------------------------

#' Run LPPLS calibrations over a grid of observation windows.
#'
#' For every combination of \code{t1} and \code{t2} (cross-join filtered by
#' \code{min_window}), calibrates the LPPLS model using the chosen
#' \code{method} and collects the estimated \code{tc}. Invalid fits (as
#' determined by \code{\link{lppls_filter}}) are discarded. Execution is
#' parallelised with \code{future.apply::future_lapply}.
#'
#' @param obs_dt A \code{data.table} with columns \code{t} (numeric time
#'   index) and \code{value} (observed series).
#' @param t1_seq Numeric vector of candidate window start times.
#' @param t2_seq Numeric vector of candidate window end times.
#' @param method One of \code{"lm"}, \code{"mlnn"}, or \code{"plnn"}.
#' @param model A trained P-LNN model object (required when
#'   \code{method = "plnn"}). Ignored otherwise.
#' @param n_starts Number of random starts for LM calibration (default 25).
#' @param min_window Minimum window length in observations (default 50).
#' @param m_range,omega_range Bounds passed to \code{\link{lppls_fit_lm}} and
#'   \code{\link{lppls_filter}}.
#' @return A \code{data.table} with columns \code{t1, t2, tc, m, omega, sse,
#'   method}. May have zero rows if no fits pass the filter.
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

#' Compute a kernel density estimate (PDF) for predicted tc values.
#'
#' Wraps \code{stats::density()} and returns the result as a
#' \code{data.table}.
#'
#' @param tc_values Numeric vector of critical-time estimates from
#'   \code{\link{lppls_multi_window}}.
#' @param bandwidth Bandwidth selector passed to \code{density()}. Default
#'   \code{"SJ"} (Sheather-Jones); falls back to \code{"nrd0"} on error.
#' @param n_points Number of equally-spaced evaluation points (default 512).
#' @return A \code{data.table} with columns \code{tc} and \code{density}.
#'   For a single input value, returns a one-row table with
#'   \code{density = 1}.
lppls_tc_pdf <- function(tc_values, bandwidth = "SJ", n_points = 512) {
  if (length(tc_values) < 2) {
    return(data.table(tc = tc_values, density = 1))
  }
  bw <- tryCatch(bandwidth, error = function(e) "nrd0")
  d <- density(tc_values, bw = bw, n = n_points)
  data.table(tc = d$x, density = d$y)
}

# --- CDF computation -------------------------------------------------------

#' Compute the empirical CDF for predicted tc values.
#'
#' @param tc_values Numeric vector of critical-time estimates.
#' @return A \code{data.table} with columns \code{tc} (sorted) and \code{cdf}
#'   (cumulative probability, \code{i/n}). Returns an empty table for
#'   zero-length input.
lppls_tc_cdf <- function(tc_values) {
  if (length(tc_values) < 1) {
    return(data.table(tc = numeric(0), cdf = numeric(0)))
  }
  tc_sorted <- sort(tc_values)
  n <- length(tc_sorted)
  data.table(tc = tc_sorted, cdf = seq_len(n) / n)
}
