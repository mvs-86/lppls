# p_lnn.R -- Poly-LPPLS Neural Network (P-LNN)
#
# Supervised model trained on large synthetic LPPLS datasets.
# Directly predicts nonlinear parameters (tc, m, omega) from a fixed-length
# time series input (default 252 points).
#
# Architecture (Eq. 3): 4 hidden layers (252 nodes each) with ReLU + output (3).
# Three variants differ only in training data noise:
#   P-LNN-WHITE, P-LNN-AR1, P-LNN-BOTH

library(torch)
library(data.table)

# --- P-LNN torch module ----------------------------------------------------

plnn_module <- nn_module(
  "PLNN",
  initialize = function(input_dim = 252L, hidden_dim = 252L) {
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, hidden_dim)
    self$fc3 <- nn_linear(hidden_dim, hidden_dim)
    self$fc4 <- nn_linear(hidden_dim, hidden_dim)
    self$out <- nn_linear(hidden_dim, 3L)
  },
  forward = function(x) {
    x %>%
      self$fc1() %>% nnf_relu() %>%
      self$fc2() %>% nnf_relu() %>%
      self$fc3() %>% nnf_relu() %>%
      self$fc4() %>% nnf_relu() %>%
      self$out()
  }
)

# --- torch dataset ----------------------------------------------------------

plnn_dataset <- dataset(
  "PLNNDataset",
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

# --- Training function ------------------------------------------------------

plnn_train <- function(X, Y,
                       batch_size = 8L,
                       lr = 1e-5,
                       epochs = 20L,
                       val_frac = 0.25,
                       seed = 42) {
  set.seed(seed)
  torch_manual_seed(seed)

  n <- nrow(X)
  n_val <- round(n * val_frac)
  n_train <- n - n_val

  idx <- sample(n)
  train_idx <- idx[seq_len(n_train)]
  val_idx <- idx[(n_train + 1):n]

  train_ds <- plnn_dataset(X[train_idx, , drop = FALSE], Y[train_idx, , drop = FALSE])
  val_ds <- plnn_dataset(X[val_idx, , drop = FALSE], Y[val_idx, , drop = FALSE])

  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

  input_dim <- ncol(X)
  model <- plnn_module(input_dim = input_dim, hidden_dim = input_dim)
  optimizer <- optim_adam(model$parameters, lr = lr)

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

    if (val_loss_history[epoch] < best_val_loss) {
      best_val_loss <- val_loss_history[epoch]
      best_state <- lapply(model$parameters, function(p) p$clone())
    }

    cat(sprintf("Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f\n",
                epoch, epochs,
                train_loss_history[epoch], val_loss_history[epoch]))
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

# --- Prediction function ----------------------------------------------------

plnn_predict <- function(model, series_vector, t_len = 252, tc_max = 302) {
  model$eval()
  scaled <- (series_vector - min(series_vector)) /
    (max(series_vector) - min(series_vector) + 1e-12)

  x <- torch_tensor(scaled, dtype = torch_float())$unsqueeze(1L)

  with_no_grad({
    raw <- model(x)
  })

  vals <- as.numeric(raw)
  list(
    tc = vals[1] * tc_max,      # denormalize
    m = vals[2],
    omega = vals[3] * 13        # denormalize
  )
}

# --- Save/load helpers ------------------------------------------------------

save_plnn <- function(model, path) {
  torch_save(model, path)
  invisible(path)
}

load_plnn <- function(path, input_dim = 252L) {
  torch_load(path)
}
