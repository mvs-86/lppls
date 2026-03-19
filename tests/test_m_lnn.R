# test_m_lnn.R -- Tests for M-LNN model

library(testthat)
library(torch)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "m_lnn.R"))

test_that("mlnn_module forward pass produces correct output shape", {
  model <- mlnn_module(input_dim = 100L)
  x <- torch_randn(1, 100)
  out <- model(x)
  expect_equal(out$shape, c(1, 3))
})

test_that("torch_lppls_reconstruct returns correct shape", {
  t_tensor <- torch_tensor(seq(0, 1, length.out = 50), dtype = torch_float())
  tc <- torch_tensor(1.1, dtype = torch_float())
  m <- torch_tensor(0.5, dtype = torch_float())
  omega <- torch_tensor(8.0, dtype = torch_float())

  X <- torch_lppls_reconstruct(t_tensor, tc, m, omega)
  expect_equal(as.integer(X$shape[1]), 50)
  expect_equal(as.integer(X$shape[2]), 4)
})

test_that("mlnn_train loss decreases over training", {
  set.seed(42)
  # Generate a clean LPPLS series
  tc <- 280; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01
  t_vec <- 0:199
  obs_vec <- lppls_value(t_vec, tc, m, omega, A, B, C1, C2)
  obs_vec <- obs_vec + rnorm(length(obs_vec), 0, 0.01 * sd(obs_vec))

  result <- mlnn_train(t_vec, obs_vec, lr = 0.01, epochs = 100,
                       alpha = 10.0, verbose = FALSE)

  expect_true(is.list(result))
  expect_true("loss_history" %in% names(result))
  expect_true("tc" %in% names(result))
  expect_true("m" %in% names(result))
  expect_true("omega" %in% names(result))

  # Loss should decrease
  early_loss <- mean(result$loss_history[1:10])
  late_loss <- mean(result$loss_history[91:100])
  expect_lt(late_loss, early_loss)
})

test_that("mlnn_train params are within valid ranges", {
  set.seed(42)
  tc <- 280; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01
  t_vec <- 0:199
  obs_vec <- lppls_value(t_vec, tc, m, omega, A, B, C1, C2)

  result <- mlnn_train(t_vec, obs_vec, lr = 0.01, epochs = 200,
                       alpha = 10.0, verbose = FALSE)

  # m should be in [0.1, 0.9] due to sigmoid activation
  expect_gte(result$m, 0.05)
  expect_lte(result$m, 0.95)
  # omega should be in [6, 13]
  expect_gte(result$omega, 5.5)
  expect_lte(result$omega, 13.5)
})
