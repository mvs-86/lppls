# run_train_plnn.R -- Train P-LNN models on synthetic data
#
# Trains six P-LNN variants:
#   P-LNN-WHITE:  white noise augmentation
#   P-LNN-AR1:    AR(1) noise augmentation
#   P-LNN-BOTH:   mixed white + AR(1) noise
#   P-LNN-ARFIMA: ARFIMA long-memory noise augmentation
#   P-LNN-MSM:    MSM (Markov-Switching Multifractal) volatility clustering noise
#   P-LNN-ALL:    equal mix of white, AR(1), and ARFIMA noise
#
# Supports both local (RStudio/Rscript) and Kaggle environments.
# On Kaggle, add this repo as a dataset and set KAGGLE_REPO_INPUT below.

library(data.table)
library(ggplot2)
library(torch)
library(future)
library(future.apply)

# --- Environment detection --------------------------------------------------

IN_KAGGLE <- dir.exists("/kaggle/working")

# Kaggle dataset slug pointing to this repo (adjust to your dataset name)
KAGGLE_REPO_INPUT <- "/kaggle/input/deep-lppls-r"

if (IN_KAGGLE) {
  old_wd <- setwd(KAGGLE_REPO_INPUT)
  source("R/lppls_synthetic.R")
  source("R/p_lnn.R")
  source("R/lppls_plots.R")
  setwd(old_wd)
  OUT_DIR <- "/kaggle/working"
} else {
  root <- rprojroot::find_root(rprojroot::is_rstudio_project)
  source(file.path(root, "R", "lppls_synthetic.R"))
  source(file.path(root, "R", "p_lnn.R"))
  source(file.path(root, "R", "lppls_plots.R"))
  OUT_DIR <- file.path(root, "output")
}

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

plan(multisession, workers = parallelly::availableCores(omit = 1))

# --- Configuration ----------------------------------------------------------

N_SAMPLES  <- 100000L
T_LEN      <- 252L
BATCH_SIZE <- 8L
LR         <- 1e-5
EPOCHS     <- 20L
SEED       <- 42

# --- Generate datasets and train --------------------------------------------

noise_types <- c("white", "ar1", "both", "arfima", "msm", "all")

for (noise in noise_types) {
  cat(sprintf("\n=== Training P-LNN-%s (%d samples) ===\n", toupper(noise), N_SAMPLES))

  cache_file <- file.path(OUT_DIR, sprintf("plnn_data_%s.rds", noise))
  if (file.exists(cache_file)) {
    cat("Loading cached dataset...\n")
    ds <- readRDS(cache_file)
  } else {
    cat("Generating synthetic dataset (parallel)...\n")
    ds <- generate_training_dataset_parallel(
      n = N_SAMPLES, t_len = T_LEN,
      noise_type = noise, seed = SEED
    )
    saveRDS(ds, cache_file)
  }

  cat(sprintf("Dataset: X[%d x %d], Y[%d x %d]\n",
              nrow(ds$X), ncol(ds$X), nrow(ds$Y), ncol(ds$Y)))

  result <- plnn_train(
    ds$X, ds$Y,
    batch_size = BATCH_SIZE, lr = LR,
    epochs = EPOCHS, val_frac = 0.25, seed = SEED
  )

  model_path <- file.path(OUT_DIR, sprintf("plnn_%s.pt", noise))
  torch_save(result$model, model_path)
  cat(sprintf("Model saved to %s\n", model_path))

  p_loss <- plot_training_loss(
    result$train_loss, result$val_loss,
    title = sprintf("P-LNN-%s: Training and Validation Loss per Epoch", toupper(noise))
  )
  ggsave(
    file.path(OUT_DIR, sprintf("plnn_%s_loss.png", noise)),
    p_loss, width = 8, height = 5, dpi = 150
  )

  p_samples <- plot_synthetic_samples(
    ds$X, ds$params, n_show = 8,
    title = sprintf("Synthetic Training Data (%s noise)", noise)
  )
  ggsave(
    file.path(OUT_DIR, sprintf("plnn_%s_samples.png", noise)),
    p_samples, width = 12, height = 6, dpi = 150
  )
}

plan(sequential)
cat("\nAll P-LNN models trained successfully.\n")
