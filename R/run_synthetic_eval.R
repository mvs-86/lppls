# run_synthetic_eval.R -- Evaluate all calibration methods on synthetic data
#
# Reproduces Figure 3 from the paper: CDF of absolute parameter errors
# for LM, M-LNN, and P-LNN across 250 test scenarios.

library(data.table)
library(ggplot2)
library(patchwork)
library(torch)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))
source(file.path(root, "R", "lppls_synthetic.R"))
source(file.path(root, "R", "m_lnn.R"))
source(file.path(root, "R", "p_lnn.R"))
source(file.path(root, "R", "lppls_plots.R"))

dir.create(file.path(root, "output"), showWarnings = FALSE, recursive = TRUE)

# --- Configuration ----------------------------------------------------------

N_TEST <- 250
T_LEN <- 252L
NOISE_TYPE <- "white"
SEED <- 123

# --- Generate test data -----------------------------------------------------

set.seed(SEED)
cat("Generating test data...\n")
test_params <- generate_lppls_params(N_TEST, t_len = T_LEN)

# --- Load P-LNN model (if available) ---------------------------------------

plnn_model <- tryCatch({
  torch_load(file.path(root, "output", "plnn_white.pt"))
}, error = function(e) {
  cat("P-LNN model not found. Train it first with run_train_plnn.R\n")
  NULL
})

# --- Evaluate each test series ----------------------------------------------

results <- list()

for (i in seq_len(N_TEST)) {
  if (i %% 50 == 0) cat(sprintf("Processing %d/%d...\n", i, N_TEST))

  p <- test_params[i]
  series <- generate_lppls_series(p, T_LEN)
  noisy <- add_white_noise(series$value, runif(1, 0.01, 0.15))
  t_vec <- series$t

  true_tc <- p$tc; true_m <- p$m; true_omega <- p$omega

  # --- LM calibration ---
  lm_fit <- tryCatch(
    lppls_fit_lm(t_vec, noisy,
      tc_range = c(T_LEN + 1, T_LEN + 60),
      n_starts = 15
    ),
    error = function(e) NULL
  )

  if (!is.null(lm_fit)) {
    lm_mse <- mean((noisy - lm_fit$fitted_values)^2)
    results <- c(results, list(data.table(
      scenario = i, method = "LM",
      tc_error = abs(lm_fit$tc - true_tc),
      m_error = abs(lm_fit$m - true_m),
      omega_error = abs(lm_fit$omega - true_omega),
      mse_error = lm_mse
    )))
  }

  # --- M-LNN calibration ---
  mlnn_fit <- tryCatch(
    mlnn_train(t_vec, noisy, lr = 0.01, epochs = 500,
               alpha = 10.0, verbose = FALSE),
    error = function(e) NULL
  )

  if (!is.null(mlnn_fit)) {
    mlnn_mse <- mean((noisy - mlnn_fit$fitted_values)^2)
    results <- c(results, list(data.table(
      scenario = i, method = "M-LNN",
      tc_error = abs(mlnn_fit$tc - true_tc),
      m_error = abs(mlnn_fit$m - true_m),
      omega_error = abs(mlnn_fit$omega - true_omega),
      mse_error = mlnn_mse
    )))
  }

  # --- P-LNN calibration ---
  if (!is.null(plnn_model)) {
    plnn_pred <- tryCatch(
      plnn_predict(plnn_model, noisy, t_len = T_LEN),
      error = function(e) NULL
    )

    if (!is.null(plnn_pred)) {
      # Reconstruct LPPLS with predicted params to compute MSE
      lin <- tryCatch(
        lppls_solve_linear(t_vec, noisy, plnn_pred$tc, plnn_pred$m, plnn_pred$omega),
        error = function(e) rep(NA_real_, 4)
      )
      if (!anyNA(lin)) {
        plnn_fitted <- lppls_value(t_vec, plnn_pred$tc, plnn_pred$m, plnn_pred$omega,
                                   lin["A"], lin["B"], lin["C1"], lin["C2"])
        plnn_mse <- mean((noisy - plnn_fitted)^2)
      } else {
        plnn_mse <- NA_real_
      }

      results <- c(results, list(data.table(
        scenario = i, method = "P-LNN-100K",
        tc_error = abs(plnn_pred$tc - true_tc),
        m_error = abs(plnn_pred$m - true_m),
        omega_error = abs(plnn_pred$omega - true_omega),
        mse_error = plnn_mse
      )))
    }
  }
}

results_dt <- rbindlist(results)

# --- Plot CDFs (Figure 3 style) ---------------------------------------------

cat("Generating error CDF plots...\n")

p_panel <- plot_error_cdf_panel(results_dt,
  title = "CDF of Absolute Parameter Error by Method"
)

ggsave(
  file.path(root, "output", "error_cdf_panel.png"),
  p_panel, width = 14, height = 10, dpi = 150
)

# Individual CDF plots
for (param in c("tc_error", "m_error", "omega_error", "mse_error")) {
  p <- plot_error_cdf(results_dt, param,
    title = sprintf("CDF of %s", param))
  ggsave(
    file.path(root, "output", sprintf("cdf_%s.png", param)),
    p, width = 7, height = 5, dpi = 150
  )
}

cat("Synthetic evaluation complete. Results saved to output/\n")
cat(sprintf("Total results: %d rows\n", nrow(results_dt)))

# Summary statistics
cat("\nMedian absolute errors by method:\n")
print(results_dt[, .(
  med_tc = median(tc_error, na.rm = TRUE),
  med_m = median(m_error, na.rm = TRUE),
  med_omega = median(omega_error, na.rm = TRUE),
  med_mse = median(mse_error, na.rm = TRUE)
), by = method])
