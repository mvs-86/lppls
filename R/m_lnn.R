# m_lnn.R -- Mono-LPPLS Neural Network (M-LNN)
#
# Physics-informed NN trained on a single time series.
# The network outputs 3 nonlinear LPPLS parameters (tc, m, omega).
# Linear parameters (A, B, C1, C2) are solved analytically inside the loss.
# Loss = MSE(observed, LPPLS_reconstructed) + alpha * boundary_penalty
#
# Architecture (Eq. 2): 2 hidden layers with ReLU, 3-node output.

library(torch)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

# --- M-LNN torch module ----------------------------------------------------

mlnn_module <- nn_module(
  "MLNN",
  initialize = function(input_dim, hidden_dim = 64) {
    self$fc1 <- nn_linear(input_dim, hidden_dim)
    self$fc2 <- nn_linear(hidden_dim, hidden_dim)
    self$out <- nn_linear(hidden_dim, 3L)
  },
  forward = function(x) {
    x %>%
      self$fc1() %>% nnf_relu() %>%
      self$fc2() %>% nnf_relu() %>%
      self$out()
  }
)

# --- Differentiable LPPLS reconstruction in torch ---------------------------

torch_lppls_reconstruct <- function(t_tensor, tc, m, omega) {
  # t_tensor: (n,) tensor of normalized time points
  # tc, m, omega: scalar tensors (differentiable)
  dt <- tc - t_tensor
  dt <- torch_clamp(dt, min = 1e-8)
  dt_m <- torch_pow(dt, m)
  log_dt <- torch_log(dt)
  f <- dt_m
  g <- dt_m * torch_cos(omega * log_dt)
  h <- dt_m * torch_sin(omega * log_dt)
  ones <- torch_ones_like(f)

  # Design matrix: (n, 4)
  X <- torch_stack(list(ones, f, g, h), dim = 2L)
  X
}

# --- Training function ------------------------------------------------------

mlnn_train <- function(t_vec, obs_vec,
                       lr = 0.01,
                       epochs = 1000,
                       alpha = 10.0,
                       hidden_dim = 64,
                       verbose = TRUE) {
  n <- length(obs_vec)

  # Min-max scale observations to [0, 1]
  obs_min <- min(obs_vec)
  obs_max <- max(obs_vec)
  obs_scaled <- (obs_vec - obs_min) / (obs_max - obs_min + 1e-12)

  # Normalize time to [0, 1]
  t_min <- min(t_vec)
  t_max <- max(t_vec)
  t_norm <- (t_vec - t_min) / (t_max - t_min + 1e-12)
  t2_norm <- 1.0  # end of series in normalized time

  t_tensor <- torch_tensor(t_norm, dtype = torch_float())
  obs_tensor <- torch_tensor(obs_scaled, dtype = torch_float())$unsqueeze(2L)

  model <- mlnn_module(n, hidden_dim)
  optimizer <- optim_adam(model$parameters, lr = lr)

  loss_history <- numeric(epochs)
  best_loss <- Inf
  best_state <- NULL

  for (epoch in seq_len(epochs)) {
    optimizer$zero_grad()

    # Forward: get raw parameter outputs
    raw_params <- model(obs_tensor$squeeze(2L)$unsqueeze(1L))
    tc_raw <- raw_params[1, 1]
    m_raw <- raw_params[1, 2]
    omega_raw <- raw_params[1, 3]

    # Map to valid ranges via sigmoid
    tc_pred <- t2_norm * (0.8 + 0.4 * torch_sigmoid(tc_raw))    # [0.8, 1.2]
    m_pred <- 0.1 + 0.8 * torch_sigmoid(m_raw)                   # [0.1, 0.9]
    omega_pred <- 6.0 + 7.0 * torch_sigmoid(omega_raw)           # [6, 13]

    # Build design matrix and solve linear params via least squares
    X_design <- torch_lppls_reconstruct(t_tensor, tc_pred, m_pred, omega_pred)

    # Solve: (X'X)^-1 X'y via lstsq
    result <- linalg_lstsq(X_design, obs_tensor)
    beta <- result[[1]]

    # Reconstruct LPPLS series
    pred <- torch_matmul(X_design, beta)

    # MSE loss
    mse_loss <- torch_mean((pred - obs_tensor)^2)

    # Boundary penalty (soft)
    penalty <- alpha * (
      torch_relu(0.8 * t2_norm - tc_pred)^2 +
      torch_relu(tc_pred - 1.2 * t2_norm)^2 +
      torch_relu(0.1 - m_pred)^2 +
      torch_relu(m_pred - 0.9)^2 +
      torch_relu(6.0 - omega_pred)^2 +
      torch_relu(omega_pred - 13.0)^2
    )

    total_loss <- mse_loss + penalty
    total_loss$backward()
    optimizer$step()

    loss_val <- total_loss$item()
    loss_history[epoch] <- loss_val

    if (loss_val < best_loss) {
      best_loss <- loss_val
      best_state <- lapply(model$parameters, function(p) p$clone())
    }

    if (verbose && epoch %% 100 == 0) {
      cat(sprintf("Epoch %d/%d | Loss: %.6f | tc: %.4f | m: %.4f | omega: %.4f\n",
                  epoch, epochs, loss_val,
                  tc_pred$item(), m_pred$item(), omega_pred$item()))
    }
  }

  # Restore best model state
  if (!is.null(best_state)) {
    for (nm in names(best_state)) {
      model$parameters[[nm]]$set_data(best_state[[nm]])
    }
  }

  # Extract final parameters
  with_no_grad({
    raw_params <- model(obs_tensor$squeeze(2L)$unsqueeze(1L))
    tc_final <- (t2_norm * (0.8 + 0.4 * torch_sigmoid(raw_params[1, 1])))$item()
    m_final <- (0.1 + 0.8 * torch_sigmoid(raw_params[1, 2]))$item()
    omega_final <- (6.0 + 7.0 * torch_sigmoid(raw_params[1, 3]))$item()
  })

  # Denormalize tc back to original time scale
  tc_real <- tc_final * (t_max - t_min) + t_min

  # Solve linear params on original scale
  lin <- lppls_solve_linear(t_vec, obs_vec, tc_real, m_final, omega_final)
  fitted <- lppls_value(t_vec, tc_real, m_final, omega_final,
                        lin["A"], lin["B"], lin["C1"], lin["C2"])

  list(
    tc = tc_real, m = m_final, omega = omega_final,
    A = lin["A"], B = lin["B"], C1 = lin["C1"], C2 = lin["C2"],
    loss_history = loss_history,
    fitted_values = fitted,
    model = model
  )
}
