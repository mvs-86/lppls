# test_glppls_synthetic.R -- Tests for G-LPPLS-NN synthetic data generation

library(testthat)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "glppls_synthetic.R"))

test_that("generate_glppls_labels produces correct shape and ranges", {
  set.seed(1)
  labels <- generate_glppls_labels(100, t_len = 252L)

  expect_equal(nrow(labels), 100)
  expect_true(all(c("mu_tc", "sigma_tc", "mu_A", "sigma_A") %in% names(labels)))
  expect_true(all(labels$mu_tc >= 252 & labels$mu_tc <= 302))
  expect_true(all(labels$sigma_tc >= 1 & labels$sigma_tc <= 14))
})

test_that("generate_params_from_label produces correct structure", {
  set.seed(1)
  labels <- generate_glppls_labels(1, t_len = 100L)
  params <- generate_params_from_label(labels[1], n_sets = 5, order = 2L,
                                        t_len = 100L)

  expect_equal(nrow(params), 5)
  expect_true("delta_t" %in% names(params))
  expect_true("delta_omega" %in% names(params))
  expect_true(all(params$alpha >= 0.01 & params$alpha <= 0.99))
  expect_true(all(params$omega >= 4.9 & params$omega <= 15))
})

test_that("rgpd_inline produces finite values", {
  set.seed(1)
  vals <- rgpd_inline(1000, xi = 0.5, sigma = 0.1)
  expect_length(vals, 1000)
  expect_true(all(is.finite(vals)))
  expect_true(all(vals >= 0))
})

test_that("add_ar1_gpd_noise produces finite values", {
  set.seed(1)
  clean <- sin(seq(0, 4 * pi, length.out = 100)) + 5
  noisy <- add_ar1_gpd_noise(clean, phi = 0.9, delta = 0.01,
                             xi = 0.5, sigma_gpd = 0.01)
  expect_length(noisy, 100)
  expect_true(all(is.finite(noisy)))
})

test_that("apply_jitter returns correct length", {
  set.seed(1)
  series <- rnorm(252)
  jittered <- apply_jitter(series, t_len = 252L)
  expect_length(jittered, 252)
  expect_true(all(is.finite(jittered)))
})

test_that("generate_glppls_dataset produces correct shapes", {
  ds <- generate_glppls_dataset(n_labels = 20, t_len = 100, n_sets = 5,
                                seed = 42)

  expect_equal(nrow(ds$X), 100)  # 20 * 5

expect_equal(ncol(ds$X), 100)
  expect_equal(nrow(ds$Y), 100)
  expect_equal(ncol(ds$Y), 4)
  expect_equal(nrow(ds$labels), 20)
})

test_that("X is scaled to [0, 1]", {
  ds <- generate_glppls_dataset(n_labels = 10, t_len = 50, n_sets = 3,
                                seed = 42)

  expect_true(all(ds$X >= 0 & ds$X <= 1))
})

test_that("Y has 4 columns for distributional labels", {
  ds <- generate_glppls_dataset(n_labels = 10, t_len = 50, n_sets = 3,
                                seed = 42)

  expect_equal(ncol(ds$Y), 4)
  expect_true(all(is.finite(ds$Y)))
})
