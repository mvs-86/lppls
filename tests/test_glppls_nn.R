# test_glppls_nn.R -- Tests for G-LPPLS-NN model

library(testthat)
library(torch)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "glppls_nn.R"))
source(file.path(root, "R", "lppls_synthetic.R"))  # for minmax_scale

test_that("glppls_module forward pass produces [batch, 4]", {
  model <- glppls_module(input_dim = 100L, hidden_dim = 100L)
  x <- torch_randn(8, 100)
  out <- model(x)
  expect_equal(out$shape, c(8, 4))
})

test_that("sigma outputs are positive (softplus)", {
  model <- glppls_module(input_dim = 100L, hidden_dim = 100L)
  x <- torch_randn(16, 100)
  out <- model(x)
  sigma_vals <- as.matrix(out[, 3:4])
  expect_true(all(sigma_vals > 0))
})

test_that("glppls_dataset has correct length and item structure", {
  X <- matrix(rnorm(50 * 100), nrow = 50, ncol = 100)
  Y <- matrix(rnorm(50 * 4), nrow = 50, ncol = 4)
  ds <- glppls_dataset(X, Y)

  expect_equal(ds$.length(), 50)
  item <- ds$.getitem(1)
  expect_true(is.list(item))
  expect_equal(as.integer(item$x$shape), 100)
  expect_equal(as.integer(item$y$shape), 4)
})

test_that("glppls_train loss decreases on small dataset", {
  set.seed(42)
  n <- 100
  t_len <- 50
  X <- matrix(runif(n * t_len), nrow = n, ncol = t_len)
  Y <- matrix(runif(n * 4), nrow = n, ncol = 4)

  result <- glppls_train(X, Y, batch_size = 16L, lr = 1e-3,
                         epochs = 5L, patience = 10L, val_frac = 0.2)

  expect_true(is.list(result))
  expect_true("train_loss" %in% names(result))
  expect_true("val_loss" %in% names(result))
  expect_lt(result$train_loss[5], result$train_loss[1])
})

test_that("glppls_predict returns 4 named components", {
  model <- glppls_module(input_dim = 100L, hidden_dim = 100L)
  series <- rnorm(100)
  pred <- glppls_predict(model, series, t_len = 100)

  expect_true(is.list(pred))
  expect_true(all(c("mu_tc", "mu_A", "sigma_tc", "sigma_A") %in% names(pred)))
  expect_length(pred$mu_tc, 1)
  expect_length(pred$sigma_tc, 1)
})

test_that("glppls_predict handles different input lengths", {
  model <- glppls_module(input_dim = 100L, hidden_dim = 100L)
  series <- rnorm(200)  # different from t_len
  pred <- glppls_predict(model, series, t_len = 100)

  expect_true(is.list(pred))
  expect_length(pred$mu_tc, 1)
})

test_that("augment_real_series produces correct shape", {
  series <- sin(seq(0, 4 * pi, length.out = 300))
  aug <- augment_real_series(series, n_augment = 5, t_len = 100)

  expect_equal(dim(aug), c(5, 100))
  expect_true(all(is.finite(aug)))
})

test_that("fine-tune freezes shared and std parameters", {
  model <- glppls_module(input_dim = 50L, hidden_dim = 50L)

  n <- 30
  X <- matrix(runif(n * 50), nrow = n, ncol = 50)
  Y <- matrix(runif(n * 4), nrow = n, ncol = 4)

  # Run finetune (will freeze/unfreeze internally)
  result <- glppls_finetune(model, X, Y, batch_size = 8L, lr = 1e-3,
                            epochs = 2L, val_frac = 0.2)

  # After finetune, all params should be unfrozen again
  for (p in result$model$parameters) {
    expect_true(p$requires_grad)
  }
})

test_that("save and load roundtrip works", {
  model <- glppls_module(input_dim = 50L, hidden_dim = 50L)
  x <- torch_randn(2, 50)

  with_no_grad({
    out1 <- as.matrix(model(x))
  })

  tmp <- tempfile(fileext = ".pt")
  save_glppls(model, tmp)
  loaded <- load_glppls(tmp)

  with_no_grad({
    out2 <- as.matrix(loaded(x))
  })

  expect_equal(out1, out2, tolerance = 1e-6)
  unlink(tmp)
})
