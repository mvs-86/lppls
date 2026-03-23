# run_glppls.R -- G-LPPLS-NN training, simulation evaluation, and empirical analysis
#
# Reproduces the pipeline from Ma & Li (2024):
#   1. Generate synthetic dataset with mixed-order LPPLS + AR(1)+GPD noise
#   2. Train G-LPPLS-NN
#   3. Evaluate on simulation test data
#   4. Apply to empirical bubble episodes

library(data.table)
library(ggplot2)
library(patchwork)
library(future)
library(future.apply)

plan(multisession, workers = parallelly::availableCores(omit = 1))

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))
source(file.path(root, "R", "lppls_higher_order.R"))
source(file.path(root, "R", "lppls_synthetic.R"))
source(file.path(root, "R", "glppls_synthetic.R"))
source(file.path(root, "R", "glppls_nn.R"))
source(file.path(root, "R", "lppls_sf.R"))
source(file.path(root, "R", "glppls_plots.R"))
source(file.path(root, "R", "lppls_plots.R"))

dir.create(file.path(root, "output"), showWarnings = FALSE, recursive = TRUE)

# --- Config -------------------------------------------------------------------

N_LABELS <- 12000L   # → ~120K sequences
T_LEN <- 252L
N_SETS <- 10L
BATCH_SIZE <- 64L
LR <- 5e-6
EPOCHS <- 30L
SEED <- 42

# =============================================================================
# 1. Generate dataset (or load cached)
# =============================================================================

cache_file <- file.path(root, "output", "glppls_train_data.rds")

if (file.exists(cache_file)) {
  cat("Loading cached training data...\n")
  ds <- readRDS(cache_file)
} else {
  cat("Generating G-LPPLS-NN training dataset...\n")
  ds <- generate_glppls_dataset_parallel(
    n_labels = N_LABELS, t_len = T_LEN, n_sets = N_SETS, seed = SEED
  )
  saveRDS(ds, cache_file)
  cat(sprintf("Saved %d training sequences to %s\n", nrow(ds$X), cache_file))
}

cat(sprintf("Dataset: X = %d × %d, Y = %d × %d\n",
            nrow(ds$X), ncol(ds$X), nrow(ds$Y), ncol(ds$Y)))

# =============================================================================
# 2. Train G-LPPLS-NN
# =============================================================================

model_file <- file.path(root, "output", "glppls_nn.pt")

if (file.exists(model_file)) {
  cat("Loading cached G-LPPLS-NN model...\n")
  model <- load_glppls(model_file)
  train_result <- NULL
} else {
  cat("Training G-LPPLS-NN...\n")
  train_result <- glppls_train(
    ds$X, ds$Y,
    batch_size = BATCH_SIZE, lr = LR, epochs = EPOCHS, seed = SEED
  )
  model <- train_result$model
  save_glppls(model, model_file)
  cat(sprintf("Model saved to %s\n", model_file))

  # Plot training loss
  p_loss <- plot_training_loss(train_result$train_loss, train_result$val_loss,
                               "G-LPPLS-NN Training Loss")
  ggsave(file.path(root, "output", "glppls_loss.png"), p_loss,
         width = 8, height = 5, dpi = 150)
}

# =============================================================================
# 3. Simulation evaluation (Section 3)
# =============================================================================

cat("Running simulation evaluation...\n")

N_TEST <- 1000L
set.seed(SEED + 1)

sim_results <- rbindlist(lapply(1:3, function(ord) {
  cat(sprintf("  Evaluating order %d...\n", ord))

  # Generate test labels
  test_labels <- generate_glppls_labels(N_TEST, T_LEN)

  results_list <- lapply(seq_len(N_TEST), function(i) {
    label <- test_labels[i]
    params <- generate_params_from_label(label, n_sets = 1, order = ord,
                                          t_len = T_LEN)

    vals <- tryCatch(
      generate_glppls_series(params[1], T_LEN),
      error = function(e) rep(NA_real_, T_LEN)
    )

    if (any(!is.finite(vals))) return(NULL)

    noisy <- add_ar1_gpd_noise(vals, phi = 0.9, delta = 0.01 * sd(vals),
                               xi = 0.5, sigma_gpd = 0.01 * sd(vals))
    scaled <- minmax_scale(noisy)

    # G-LPPLS-NN prediction
    nn_pred <- tryCatch(
      glppls_predict(model, noisy, T_LEN),
      error = function(e) list(mu_tc = NA, mu_A = NA,
                               sigma_tc = NA, sigma_A = NA)
    )

    # LPPLS-RF (LM) prediction -- get tc and A from multi-start
    t_seq <- seq(0, T_LEN - 1)
    lm_fit <- tryCatch(
      lppls_fit_lm(t_seq, noisy, n_starts = 10),
      error = function(e) NULL
    )

    lm_tc <- if (!is.null(lm_fit)) lm_fit$tc else NA_real_
    lm_A <- if (!is.null(lm_fit)) lm_fit$A else NA_real_

    # LPPLS-SF prediction
    sf_fit <- tryCatch(
      lppls_sf_fit(t_seq, noisy, n_starts = 10, max_iter = 100),
      error = function(e) NULL
    )

    sf_tc <- if (!is.null(sf_fit)) sf_fit$tc else NA_real_
    sf_A <- if (!is.null(sf_fit)) sf_fit$A else NA_real_

    data.table(
      order = ord,
      true_mu_tc = label$mu_tc,
      true_sigma_tc = label$sigma_tc,
      true_mu_A = label$mu_A,
      true_sigma_A = label$sigma_A,

      # G-LPPLS-NN errors
      method = c("G-LPPLS-NN", "LPPLS-RF", "LPPLS-SF"),
      error_mu_tc = c(
        nn_pred$mu_tc - label$mu_tc,
        lm_tc - label$mu_tc,
        sf_tc - label$mu_tc
      ),
      error_sigma_tc = c(
        nn_pred$sigma_tc - label$sigma_tc,
        NA_real_, NA_real_  # point methods have no sigma
      ),
      error_mu_A = c(
        nn_pred$mu_A - label$mu_A,
        lm_A - label$mu_A,
        sf_A - label$mu_A
      ),
      error_sigma_exp_A = c(
        exp(nn_pred$sigma_A) - exp(label$sigma_A),
        NA_real_, NA_real_
      )
    )
  })

  rbindlist(results_list[!sapply(results_list, is.null)])
}))

# Plot CDFs
p_cdf <- plot_glppls_sim_cdf(sim_results, "Simulation Error CDFs")
ggsave(file.path(root, "output", "glppls_sim_cdf.png"), p_cdf,
       width = 14, height = 10, dpi = 150)
cat("Saved simulation CDF plot\n")

# =============================================================================
# 4. Empirical analysis (Section 4)
# =============================================================================

cat("Running empirical analysis...\n")

# Helper to download and prepare data
prepare_series <- function(symbol, from, to) {
  if (!requireNamespace("quantmod", quietly = TRUE)) {
    cat("quantmod not installed, skipping empirical analysis\n")
    return(NULL)
  }
  env <- new.env()
  tryCatch({
    quantmod::getSymbols(symbol, from = from, to = to, env = env)
    xts_obj <- env[[ls(env)[1]]]
    close_col <- grep("Close", names(xts_obj), value = TRUE)[1]
    prices <- as.numeric(xts_obj[, close_col])
    dates <- as.Date(zoo::index(xts_obj))
    data.table(t = seq_along(prices), value = log(prices), date = dates)
  }, error = function(e) {
    cat(sprintf("Failed to download %s: %s\n", symbol, e$message))
    NULL
  })
}

empirical_cases <- list(
  list(symbol = "BTC-USD", from = "2020-01-01", to = "2021-06-01",
       name = "btc"),
  list(symbol = "^GSPC",   from = "2019-01-01", to = "2020-04-01",
       name = "sp500"),
  list(symbol = "CRB",     from = "2007-01-01", to = "2008-09-01",
       name = "crb")
)

for (case in empirical_cases) {
  obs_dt <- prepare_series(case$symbol, case$from, case$to)
  if (is.null(obs_dt)) next

  n <- nrow(obs_dt)
  cutoff <- round(0.9 * n)
  input_series <- obs_dt$value[seq_len(cutoff)]

  # G-LPPLS-NN
  nn_pred <- glppls_predict(model, input_series, T_LEN)

  # G-LPPLS-NN with fine-tuning
  aug_X <- augment_real_series(input_series, n_augment = 10, t_len = T_LEN)
  aug_X_scaled <- t(apply(aug_X, 1, minmax_scale))

  # Create rough labels from base prediction for fine-tuning
  end_val <- abs(input_series[length(input_series)]) + 1e-10
  nn_Y <- matrix(
    rep(c(nn_pred$mu_tc / T_LEN, nn_pred$sigma_tc / T_LEN,
          nn_pred$mu_A / log(end_val + 1),
          exp(nn_pred$sigma_A) / (end_val + 1e-10)), 10),
    nrow = 10, ncol = 4, byrow = TRUE
  )

  tuned_model <- tryCatch({
    # Clone model for fine-tuning
    model_copy <- glppls_module(input_dim = T_LEN, hidden_dim = T_LEN)
    for (nm in names(model$parameters)) {
      model_copy$parameters[[nm]]$set_data(model$parameters[[nm]]$clone())
    }
    ft_result <- glppls_finetune(model_copy, aug_X_scaled, nn_Y,
                                 batch_size = 3L, lr = 5e-7, epochs = 3L)
    ft_result$model
  }, error = function(e) NULL)

  tuned_pred <- if (!is.null(tuned_model)) {
    glppls_predict(tuned_model, input_series, T_LEN)
  } else {
    nn_pred
  }

  # LPPLS-RF
  t_seq <- seq(0, length(input_series) - 1)
  rf_fit <- tryCatch(
    lppls_fit_lm(t_seq, input_series, n_starts = 25),
    error = function(e) NULL
  )

  # LPPLS-SF
  sf_fit <- tryCatch(
    lppls_sf_fit(t_seq, input_series, n_starts = 25),
    error = function(e) NULL
  )

  predictions <- list()
  predictions[["G-LPPLS-NN"]] <- nn_pred
  predictions[["G-LPPLS-NN-Tuned"]] <- tuned_pred

  if (!is.null(rf_fit)) {
    predictions[["LPPLS-RF"]] <- list(
      mu_tc = rf_fit$tc, mu_A = rf_fit$A,
      sigma_tc = 5, sigma_A = log(0.1)  # rough uncertainty
    )
  }
  if (!is.null(sf_fit)) {
    predictions[["LPPLS-SF"]] <- list(
      mu_tc = sf_fit$tc, mu_A = sf_fit$A,
      sigma_tc = 5, sigma_A = log(0.1)
    )
  }

  # Plot
  p <- plot_glppls_bubble_full(obs_dt, predictions,
                               title = sprintf("Bubble Detection: %s",
                                               toupper(case$name)),
                               date_col = "date")
  ggsave(file.path(root, "output", sprintf("glppls_%s.png", case$name)), p,
         width = 12, height = 8, dpi = 150)
  cat(sprintf("Saved %s plot\n", case$name))
}

plan(sequential)
cat("Done.\n")
