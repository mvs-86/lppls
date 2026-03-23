# test_synthetic.R -- Tests for synthetic LPPLS data generation

library(testthat)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_synthetic.R"))

test_that("generate_lppls_params produces correct dimensions and ranges", {
  set.seed(42)
  params <- generate_lppls_params(100, t_len = 252)
  expect_equal(nrow(params), 100)
  expect_true(all(params$tc >= 252 & params$tc <= 302))
  expect_true(all(params$m >= 0.1 & params$m <= 0.9))
  expect_true(all(params$omega >= 6 & params$omega <= 13))
  expect_true(all(params$B < 0))
})

test_that("generate_lppls_series produces correct length", {
  set.seed(42)
  params <- generate_lppls_params(1, t_len = 100)
  series <- generate_lppls_series(params[1], t_len = 100)
  expect_equal(nrow(series), 100)
  expect_true(all(c("t", "value") %in% names(series)))
})

test_that("add_white_noise adds noise with expected amplitude", {
  set.seed(42)
  vals <- sin(seq(0, 4 * pi, length.out = 200)) + 5
  amplitude <- 0.10
  noisy <- add_white_noise(vals, amplitude)
  noise <- noisy - vals
  expect_equal(sd(noise), amplitude * sd(vals), tolerance = 0.05 * sd(vals))
})

test_that("add_ar1_noise has positive lag-1 autocorrelation", {
  set.seed(42)
  vals <- sin(seq(0, 4 * pi, length.out = 500)) + 5
  noisy <- add_ar1_noise(vals, amplitude = 0.05, phi = 0.9)
  noise <- noisy - vals
  acf_val <- acf(noise, lag.max = 1, plot = FALSE)$acf[2]
  expect_gt(acf_val, 0.5)
})

test_that("generate_training_dataset returns correct shapes", {
  set.seed(42)
  ds <- generate_training_dataset(n = 50, t_len = 100, noise_type = "white")
  expect_equal(nrow(ds$X), 50)
  expect_equal(ncol(ds$X), 100)
  expect_equal(nrow(ds$Y), 50)
  expect_equal(ncol(ds$Y), 3)
  expect_true(all(ds$X >= 0 & ds$X <= 1))  # min-max scaled
})

test_that("minmax_scale maps to [0,1]", {
  x <- c(-5, 0, 3, 10)
  s <- minmax_scale(x)
  expect_equal(min(s), 0)
  expect_equal(max(s), 1)
})

test_that("add_arfima_noise amplitude calibration is approximate", {
  set.seed(42)
  vals <- sin(seq(0, 4 * pi, length.out = 500)) + 5
  amplitude <- 0.05
  noisy <- add_arfima_noise(vals, amplitude, d = 0.3)
  noise <- noisy - vals
  expect_equal(sd(noise), amplitude * sd(vals), tolerance = 0.1 * sd(vals))
})

test_that("add_arfima_noise produces long-range dependence", {
  set.seed(42)
  vals <- sin(seq(0, 4 * pi, length.out = 500)) + 5
  noisy <- add_arfima_noise(vals, amplitude = 0.05, d = 0.4)
  noise <- noisy - vals
  acf_val <- acf(noise, lag.max = 10, plot = FALSE)$acf[11]
  expect_gt(acf_val, 0.05)
})

test_that("generate_training_dataset returns correct shapes for arfima noise", {
  set.seed(42)
  ds <- generate_training_dataset(n = 20, t_len = 100, noise_type = "arfima")
  expect_equal(dim(ds$X), c(20L, 100L))
  expect_equal(dim(ds$Y), c(20L, 3L))
  expect_true(all(ds$X >= 0 & ds$X <= 1))
})

test_that("generate_training_dataset returns correct shapes for all noise", {
  set.seed(42)
  ds <- generate_training_dataset(n = 30, t_len = 100, noise_type = "all")
  expect_equal(dim(ds$X), c(30L, 100L))
  expect_equal(dim(ds$Y), c(30L, 3L))
  expect_true(all(ds$X >= 0 & ds$X <= 1))
})
