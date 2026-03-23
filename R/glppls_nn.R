# glppls_nn.R -- Generalized LPPLS Neural Network (G-LPPLS-NN)
#
# Dual-branch architecture from Ma & Li (2024) Figure C.1:
#   Input → Shared Linear+GELU
#            ├─ Mean Branch (6× Linear+PReLU, residual, LayerNorm) → μ_tc, μ_A
#            └─ Std Branch (2× Linear+GELU, softplus) → σ_tc, σ_A
#   Output: [batch, 4]

library(torch)
library(data.table)

# --- G-LPPLS-NN torch module -------------------------------------------------

#' G-LPPLS-NN module.
#'
#' Predicts distributional parameters {mu_tc, mu_A, sigma_tc, sigma_A} from a
#' fixed-length input time series using a dual-branch architecture with
#' residual connections.
#'
#' @param input_dim Integer length of input series (default 252).
#' @param hidden_dim Integer width of hidden layers (default 252).
glppls_module <- nn_module(
  "GLPPLS",
  initialize = function(input_dim = 252L, hidden_dim = 252L) {
    # Shared layer
    self$shared <- nn_linear(input_dim, hidden_dim)

    # Mean branch: 6 Linear+PReLU layers
    self$mean_fc1 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu1 <- nn_prelu()
    self$mean_fc2 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu2 <- nn_prelu()
    self$mean_fc3 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu3 <- nn_prelu()
    self$mean_fc4 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu4 <- nn_prelu()
    self$mean_fc5 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu5 <- nn_prelu()
    self$mean_fc6 <- nn_linear(hidden_dim, hidden_dim)
    self$mean_prelu6 <- nn_prelu()
    self$mean_ln <- nn_layer_norm(hidden_dim)
    self$mean_out <- nn_linear(hidden_dim, 2L)

    # Std branch: 2 Linear+GELU layers
    self$std_fc1 <- nn_linear(hidden_dim, hidden_dim)
    self$std_fc2 <- nn_linear(hidden_dim, hidden_dim)
    self$std_out <- nn_linear(hidden_dim, 2L)
  },

  forward = function(x) {
    # Shared
    shared_out <- nnf_gelu(self$shared(x))

    # Mean branch with residual from shared
    m <- self$mean_prelu1(self$mean_fc1(shared_out))
    m <- self$mean_prelu2(self$mean_fc2(m))
    m <- self$mean_prelu3(self$mean_fc3(m))
    m <- self$mean_prelu4(self$mean_fc4(m))
    m <- self$mean_prelu5(self$mean_fc5(m))
    m <- self$mean_prelu6(self$mean_fc6(m))
    m <- m + shared_out  # residual connection
    m <- self$mean_ln(m)
    mu <- self$mean_out(m)

    # Std branch
    s <- nnf_gelu(self$std_fc1(shared_out))
    s <- nnf_gelu(self$std_fc2(s))
    sigma <- nnf_softplus(self$std_out(s))  # ensure positivity

    torch_cat(list(mu, sigma), dim = 2L)  # [batch, 4]
  }
)

# --- torch dataset ------------------------------------------------------------

#' Torch dataset for G-LPPLS-NN training.
#'
#' @param X Numeric matrix (n, t_len) of min-max scaled time series.
#' @param Y Numeric matrix (n, 4) of normalised distributional labels.
glppls_dataset <- dataset(
  "GLPPLSDataset",
  initialize = function(X, Y) {
    self$X <- torch_tensor(X, dtype = torch_float())
    self$Y <- torch_tensor(Y, dtype = torch_float())
  },
  .getitem = function(i) {
    list(x = self$X[i, ], y = self$Y[i, ])
  },
  .length = function() {
    self$X$shape[1]
  }
)

# --- Training function --------------------------------------------------------

#' Train a G-LPPLS-NN model.
#'
#' Uses Adam with step-LR decay and early stopping on validation loss.
#'
#' @param X Numeric matrix (n, t_len) of min-max scaled series.
#' @param Y Numeric matrix (n, 4) of normalised distributional labels.
#' @param batch_size Mini-batch size (default 64).
#' @param lr Learning rate (default 5e-6).
#' @param epochs Max training epochs (default 30).
#' @param lr_decay LR decay factor per step (default 0.8).
#' @param lr_step Epochs per LR decay step (default 5).
#' @param patience Early stopping patience (default 1).
#' @param val_frac Validation fraction (default 0.167).
#' @param seed Random seed.
#' @return list(model, train_loss, val_loss, best_val_loss).
glppls_train <- function(X, Y,
                         batch_size = 64L,
                         lr = 5e-6,
                         epochs = 30L,
                         lr_decay = 0.8,
                         lr_step = 5L,
                         patience = 1L,
                         val_frac = 0.167,
                         seed = 42) {
  set.seed(seed)
  torch_manual_seed(seed)

  n <- nrow(X)
  n_val <- round(n * val_frac)
  n_train <- n - n_val

  idx <- sample(n)
  train_idx <- idx[seq_len(n_train)]
  val_idx <- idx[(n_train + 1):n]

  train_ds <- glppls_dataset(X[train_idx, , drop = FALSE],
                             Y[train_idx, , drop = FALSE])
  val_ds <- glppls_dataset(X[val_idx, , drop = FALSE],
                           Y[val_idx, , drop = FALSE])

  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

  input_dim <- ncol(X)
  model <- glppls_module(input_dim = input_dim, hidden_dim = input_dim)
  optimizer <- optim_adam(model$parameters, lr = lr)
  current_lr <- lr

  train_loss_history <- numeric(epochs)
  val_loss_history <- numeric(epochs)
  best_val_loss <- Inf
  best_state <- NULL
  no_improve <- 0L

  for (epoch in seq_len(epochs)) {
    # Training
    model$train()
    train_loss_sum <- 0
    train_batches <- 0

    coro::loop(for (batch in train_dl) {
      optimizer$zero_grad()
      pred <- model(batch$x)
      loss <- nnf_mse_loss(pred, batch$y)
      loss$backward()
      optimizer$step()
      train_loss_sum <- train_loss_sum + loss$item()
      train_batches <- train_batches + 1
    })

    # Validation
    model$eval()
    val_loss_sum <- 0
    val_batches <- 0

    with_no_grad({
      coro::loop(for (batch in val_dl) {
        pred <- model(batch$x)
        loss <- nnf_mse_loss(pred, batch$y)
        val_loss_sum <- val_loss_sum + loss$item()
        val_batches <- val_batches + 1
      })
    })

    train_loss_history[epoch] <- train_loss_sum / train_batches
    val_loss_history[epoch] <- val_loss_sum / val_batches

    # Step LR decay
    if (epoch %% lr_step == 0) {
      current_lr <- current_lr * lr_decay
      for (pg in seq_along(optimizer$param_groups)) {
        optimizer$param_groups[[pg]]$lr <- current_lr
      }
    }

    if (val_loss_history[epoch] < best_val_loss) {
      best_val_loss <- val_loss_history[epoch]
      best_state <- lapply(model$parameters, function(p) p$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }

    cat(sprintf("Epoch %d/%d | Train: %.6f | Val: %.6f | LR: %.2e\n",
                epoch, epochs,
                train_loss_history[epoch], val_loss_history[epoch],
                current_lr))

    if (no_improve >= patience) {
      cat(sprintf("Early stopping at epoch %d\n", epoch))
      train_loss_history <- train_loss_history[seq_len(epoch)]
      val_loss_history <- val_loss_history[seq_len(epoch)]
      break
    }
  }

  # Restore best model
  if (!is.null(best_state)) {
    for (nm in names(best_state)) {
      model$parameters[[nm]]$set_data(best_state[[nm]])
    }
  }

  list(
    model = model,
    train_loss = train_loss_history,
    val_loss = val_loss_history,
    best_val_loss = best_val_loss
  )
}

# --- Fine-tuning function -----------------------------------------------------

#' Fine-tune G-LPPLS-NN mean branch on real data.
#'
#' Freezes the shared layer and std branch; only trains mean branch parameters.
#'
#' @param model Trained glppls_module.
#' @param X Numeric matrix of augmented real series.
#' @param Y Numeric matrix of distributional labels.
#' @param batch_size Mini-batch size (default 3).
#' @param lr Learning rate (default 5e-7).
#' @param epochs Fine-tuning epochs (default 3).
#' @param val_frac Validation fraction (default 0.1).
#' @param seed Random seed.
#' @return list(model, train_loss, val_loss, best_val_loss).
glppls_finetune <- function(model, X, Y,
                            batch_size = 3L,
                            lr = 5e-7,
                            epochs = 3L,
                            val_frac = 0.1,
                            seed = 42) {
  set.seed(seed)
  torch_manual_seed(seed)

  # Freeze shared layer
  model$shared$weight$requires_grad_(FALSE)
  model$shared$bias$requires_grad_(FALSE)

  # Freeze std branch
  for (layer_name in c("std_fc1", "std_fc2", "std_out")) {
    layer <- model[[layer_name]]
    layer$weight$requires_grad_(FALSE)
    layer$bias$requires_grad_(FALSE)
  }

  # Only optimize mean branch parameters (those still requiring grad)
  trainable_params <- Filter(function(p) p$requires_grad, model$parameters)
  optimizer <- optim_adam(trainable_params, lr = lr)

  n <- nrow(X)
  n_val <- max(1L, round(n * val_frac))
  n_train <- n - n_val

  idx <- sample(n)
  train_ds <- glppls_dataset(X[idx[seq_len(n_train)], , drop = FALSE],
                             Y[idx[seq_len(n_train)], , drop = FALSE])
  val_ds <- glppls_dataset(X[idx[(n_train + 1):n], , drop = FALSE],
                           Y[idx[(n_train + 1):n], , drop = FALSE])

  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

  train_loss_history <- numeric(epochs)
  val_loss_history <- numeric(epochs)
  best_val_loss <- Inf
  best_state <- NULL

  for (epoch in seq_len(epochs)) {
    model$train()
    train_loss_sum <- 0
    train_batches <- 0

    coro::loop(for (batch in train_dl) {
      optimizer$zero_grad()
      pred <- model(batch$x)
      loss <- nnf_mse_loss(pred, batch$y)
      loss$backward()
      optimizer$step()
      train_loss_sum <- train_loss_sum + loss$item()
      train_batches <- train_batches + 1
    })

    model$eval()
    val_loss_sum <- 0
    val_batches <- 0

    with_no_grad({
      coro::loop(for (batch in val_dl) {
        pred <- model(batch$x)
        loss <- nnf_mse_loss(pred, batch$y)
        val_loss_sum <- val_loss_sum + loss$item()
        val_batches <- val_batches + 1
      })
    })

    train_loss_history[epoch] <- train_loss_sum / max(train_batches, 1)
    val_loss_history[epoch] <- val_loss_sum / max(val_batches, 1)

    if (val_loss_history[epoch] < best_val_loss) {
      best_val_loss <- val_loss_history[epoch]
      best_state <- lapply(model$parameters, function(p) p$clone())
    }

    cat(sprintf("Fine-tune %d/%d | Train: %.6f | Val: %.6f\n",
                epoch, epochs,
                train_loss_history[epoch], val_loss_history[epoch]))
  }

  if (!is.null(best_state)) {
    for (nm in names(best_state)) {
      model$parameters[[nm]]$set_data(best_state[[nm]])
    }
  }

  # Unfreeze all parameters for future use
  for (p in model$parameters) p$requires_grad_(TRUE)

  list(
    model = model,
    train_loss = train_loss_history,
    val_loss = val_loss_history,
    best_val_loss = best_val_loss
  )
}

# --- Augmentation for real data -----------------------------------------------

#' Augment a real time series by random resampling + interpolation.
#'
#' @param series Numeric vector of observed values.
#' @param n_augment Number of augmented copies (default 10).
#' @param t_len Target output length (default 252).
#' @return Matrix (n_augment, t_len) of augmented series.
augment_real_series <- function(series, n_augment = 10L, t_len = 252L) {
  n <- length(series)
  out <- matrix(0, nrow = n_augment, ncol = t_len)

  for (i in seq_len(n_augment)) {
    frac <- runif(1, 0.66, 0.99)
    n_keep <- max(2L, round(n * frac))
    keep_idx <- sort(sample(n, n_keep))
    resampled <- series[keep_idx]
    out[i, ] <- approx(seq_along(resampled), resampled, n = t_len)$y
  }
  out
}

# --- Prediction function ------------------------------------------------------

#' Predict distributional parameters from a single time series.
#'
#' @param model Trained glppls_module.
#' @param series_vector Numeric vector of observed values.
#' @param t_len Expected input length (default 252).
#' @return list(mu_tc, mu_A, sigma_tc, sigma_A) on original scales.
glppls_predict <- function(model, series_vector, t_len = 252L) {
  model$eval()

  # Interpolate to t_len if needed
  if (length(series_vector) != t_len) {
    series_vector <- approx(seq_along(series_vector), series_vector,
                            n = t_len)$y
  }

  # Min-max scale
  end_val <- abs(series_vector[t_len])
  scaled <- minmax_scale(series_vector)
  x <- torch_tensor(scaled, dtype = torch_float())$unsqueeze(1L)

  with_no_grad({
    raw <- model(x)
  })

  vals <- as.numeric(raw)

  # Denormalize
  list(
    mu_tc = vals[1] * t_len,
    mu_A = vals[2] * log(end_val + 1),
    sigma_tc = vals[3] * t_len,
    sigma_A = log(vals[4] * (end_val + 1e-10) + 1e-10)
  )
}

# --- Save/load helpers --------------------------------------------------------

#' Save a trained G-LPPLS-NN model.
#'
#' @param model A glppls_module object.
#' @param path File path (typically ending in .pt).
#' @return path (invisibly).
save_glppls <- function(model, path) {
  torch_save(model, path)
  invisible(path)
}

#' Load a previously saved G-LPPLS-NN model.
#'
#' @param path File path to the .pt file.
#' @return The loaded glppls_module.
load_glppls <- function(path) {
  torch_load(path)
}
