# run_real_data.R -- Real-data analysis with tc PDF overlay plots
#
# Downloads market data via quantmod and produces Figures 4-6 style plots:
# time series with LPPLS fits and tc PDF overlays.
#
# Datasets:
#   1. Nasdaq Composite (^IXIC) 1997-2000: Dot-com bubble
#   2. ProShares Ultra Silver (AGQ) 2010-2011: Silver bubble

library(data.table)
library(ggplot2)
library(patchwork)
library(quantmod)
library(torch)
library(future)
library(future.apply)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))
source(file.path(root, "R", "lppls_pdf.R"))
source(file.path(root, "R", "lppls_plots.R"))
source(file.path(root, "R", "m_lnn.R"))
source(file.path(root, "R", "p_lnn.R"))

dir.create(file.path(root, "output"), showWarnings = FALSE, recursive = TRUE)

plan(multisession, workers = parallelly::availableCores(omit = 1))

# --- Helper: download and prepare data --------------------------------------

download_series <- function(symbol, from, to, col = "Adjusted") {
  xts_data <- getSymbols(symbol, src = "yahoo", from = from, to = to,
                         auto.assign = FALSE)
  adj <- Ad(xts_data)
  dt <- data.table(
    date = index(adj),
    value = as.numeric(adj)
  )
  dt <- dt[!is.na(value)]
  dt[, t := seq_len(.N)]
  dt
}

# --- Helper: run multi-window and produce PDF overlay -----------------------

run_analysis <- function(obs_dt, t1_start, t2_end, dataset_name,
                         tc_actual = NULL, n_windows = 30, plnn_model = NULL) {
  n <- nrow(obs_dt)

  # Define window grid
  t1_seq <- seq(t1_start, t1_start + round(n * 0.3), length.out = n_windows)
  t2_seq <- seq(t2_end - round(n * 0.15), t2_end, length.out = n_windows)

  cat(sprintf("\n=== %s: LM multi-window calibration ===\n", dataset_name))
  lm_results <- lppls_multi_window(
    obs_dt, t1_seq, t2_seq,
    method = "lm", n_starts = 15
  )

  pdf_list <- list()

  if (nrow(lm_results) > 2) {
    pdf_list[["Levenberg-Marquardt"]] <- lppls_tc_pdf(lm_results$tc)
  }

  # M-LNN: run on the main window
  cat(sprintf("=== %s: M-LNN calibration ===\n", dataset_name))
  main_sub <- obs_dt[t >= t1_start & t <= t2_end]
  mlnn_results <- tryCatch({
    # Run M-LNN for multiple windows (subset of grid for speed)
    mlnn_tc_vals <- c()
    for (i in seq(1, min(10, length(t2_seq)))) {
      sub <- obs_dt[t >= t1_seq[1] & t <= t2_seq[i]]
      if (nrow(sub) >= 50) {
        fit <- tryCatch(
          mlnn_train(sub$t, sub$value, epochs = 500, verbose = FALSE),
          error = function(e) NULL
        )
        if (!is.null(fit) && lppls_filter(fit)) {
          mlnn_tc_vals <- c(mlnn_tc_vals, fit$tc)
        }
      }
    }
    mlnn_tc_vals
  }, error = function(e) numeric(0))

  if (length(mlnn_results) > 2) {
    pdf_list[["Mono-LPPLS-NN"]] <- lppls_tc_pdf(mlnn_results)
  }

  # P-LNN: if model available
  if (!is.null(plnn_model)) {
    cat(sprintf("=== %s: P-LNN calibration ===\n", dataset_name))
    plnn_results <- lppls_multi_window(
      obs_dt, t1_seq, t2_seq,
      method = "plnn", model = plnn_model
    )
    if (nrow(plnn_results) > 2) {
      pdf_list[["Poly-LPPLS-NN 100K"]] <- lppls_tc_pdf(plnn_results$tc)
    }
  }

  # Generate overlay plot
  if (length(pdf_list) > 0) {
    p <- plot_tc_pdf_overlay(
      obs_dt, pdf_list,
      t1_first = t1_start, t2_last = t2_end,
      tc_actual = tc_actual,
      title = dataset_name,
      date_col = if ("date" %in% names(obs_dt)) "date" else NULL
    )

    fname <- gsub("[^a-zA-Z0-9]", "_", tolower(dataset_name))
    ggsave(
      file.path(root, "output", sprintf("tc_pdf_%s.png", fname)),
      p, width = 12, height = 6, dpi = 150
    )
    cat(sprintf("Plot saved: output/tc_pdf_%s.png\n", fname))
  } else {
    cat("Not enough valid fits to produce PDF overlay.\n")
  }

  list(lm = lm_results, pdf_list = pdf_list)
}

# --- 1. Nasdaq Dot-com Bubble -----------------------------------------------

cat("\nDownloading Nasdaq Composite data...\n")
nasdaq <- tryCatch(
  download_series("^IXIC", "1997-01-01", "2001-06-30"),
  error = function(e) {
    cat("Failed to download Nasdaq data:", conditionMessage(e), "\n")
    NULL
  }
)

if (!is.null(nasdaq)) {
  # Dot-com peak was around March 10, 2000
  # Use data up to around the peak
  peak_idx <- nasdaq[, which.max(value)]
  t2_end <- min(peak_idx, nrow(nasdaq))
  t1_start <- max(1, t2_end - 500)

  run_analysis(
    nasdaq, t1_start, t2_end,
    dataset_name = "Dot-com Bubble (Nasdaq Composite)",
    tc_actual = peak_idx
  )
}

# --- 2. Silver Bubble (AGQ) -------------------------------------------------

cat("\nDownloading ProShares Ultra Silver (AGQ) data...\n")
agq <- tryCatch(
  download_series("AGQ", "2010-01-01", "2012-06-30"),
  error = function(e) {
    cat("Failed to download AGQ data:", conditionMessage(e), "\n")
    NULL
  }
)

if (!is.null(agq)) {
  peak_idx <- agq[, which.max(value)]
  t2_end <- min(peak_idx, nrow(agq))
  t1_start <- max(1, t2_end - 250)

  run_analysis(
    agq, t1_start, t2_end,
    dataset_name = "2011 Silver Bubble (AGQ)",
    tc_actual = peak_idx
  )
}

plan(sequential)
cat("\nReal data analysis complete.\n")
