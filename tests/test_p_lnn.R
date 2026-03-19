# test_p_lnn.R -- Tests for P-LNN model

library(testthat)
library(torch)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "p_lnn.R"))

test_that("plnn_module forward pass produces correct shape", {
  model <- plnn_module(input_dim = 100L, hidden_dim = 100L)
  x <- torch_randn(8, 100)
  out <- model(x)
  expect_equal(out$shape, c(8, 3))
})

test_that("plnn_dataset has correct length and item structure", {
  X <- matrix(rnorm(50 * 100), nrow = 50, ncol = 100)
  Y <- matrix(rnorm(50 * 3), nrow = 50, ncol = 3)
  ds <- plnn_dataset(X, Y)
  expect_equal(ds$.length(), 50)

  item <- ds$.getitem(1)
  expect_true(is.list(item))
  expect_equal(as.integer(item$x$shape), 100)
  expect_equal(as.integer(item$y$shape), 3)
})

test_that("plnn_train loss decreases on small dataset", {
  set.seed(42)
  source(file.path(root, "R", "lppls_synthetic.R"))

  ds <- generate_training_dataset(n = 100, t_len = 100, noise_type = "white")
  result <- plnn_train(ds$X, ds$Y, batch_size = 16L, lr = 1e-3,
                       epochs = 10L, val_frac = 0.2)

  expect_true(is.list(result))
  expect_true("train_loss" %in% names(result))
  expect_true("val_loss" %in% names(result))
  expect_lt(result$train_loss[10], result$train_loss[1])
})

test_that("plnn_predict returns correct structure", {
  model <- plnn_module(input_dim = 100L, hidden_dim = 100L)
  series <- rnorm(100)
  pred <- plnn_predict(model, series, t_len = 100, tc_max = 150)

  expect_true(is.list(pred))
  expect_true(all(c("tc", "m", "omega") %in% names(pred)))
  expect_length(pred$tc, 1)
  expect_length(pred$m, 1)
  expect_length(pred$omega, 1)
})
