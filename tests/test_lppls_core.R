# test_lppls_core.R -- Tests for LPPLS core functions

library(testthat)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

test_that("lppls_basis returns correct dimensions", {
  t <- 0:99
  X <- lppls_basis(t, tc = 120, m = 0.5, omega = 8)
  expect_equal(nrow(X), 100)
  expect_equal(ncol(X), 4)
  expect_true(all(X[, 1] == 1))  # intercept column
})

test_that("lppls_value computes correct values for known params", {
  t <- c(0, 50)
  tc <- 100; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01

  vals <- lppls_value(t, tc, m, omega, A, B, C1, C2)
  expect_length(vals, 2)

  # Manual check for t=0
  dt <- tc - 0
  f <- dt^m
  g <- f * cos(omega * log(dt))
  h <- f * sin(omega * log(dt))
  expected <- A + B * f + C1 * g + C2 * h
  expect_equal(vals[1], expected, tolerance = 1e-10)
})

test_that("lppls_solve_linear recovers exact params from noiseless data", {
  tc <- 300; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01

  t <- 0:199
  obs <- lppls_value(t, tc, m, omega, A, B, C1, C2)

  lin <- lppls_solve_linear(t, obs, tc, m, omega)
  expect_equal(unname(lin["A"]), A, tolerance = 1e-8)
  expect_equal(unname(lin["B"]), B, tolerance = 1e-8)
  expect_equal(unname(lin["C1"]), C1, tolerance = 1e-8)
  expect_equal(unname(lin["C2"]), C2, tolerance = 1e-8)
})

test_that("lppls_residuals are zero for exact data", {
  tc <- 300; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01

  t <- 0:199
  obs <- lppls_value(t, tc, m, omega, A, B, C1, C2)
  resid <- lppls_residuals(t, obs, tc, m, omega)
  expect_true(all(abs(resid) < 1e-8))
})

test_that("lppls_sse is zero for exact data", {
  tc <- 300; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01

  t <- 0:199
  obs <- lppls_value(t, tc, m, omega, A, B, C1, C2)
  sse <- lppls_sse(t, obs, tc, m, omega)
  expect_lt(sse, 1e-14)
})

test_that("lppls_fit_lm recovers params from low-noise synthetic data", {
  set.seed(123)
  tc <- 280; m <- 0.5; omega <- 8
  A <- 10; B <- -0.5; C1 <- 0.02; C2 <- 0.01

  t <- 0:249
  clean <- lppls_value(t, tc, m, omega, A, B, C1, C2)
  obs <- clean + rnorm(length(clean), 0, 0.005 * sd(clean))

  fit <- lppls_fit_lm(t, obs,
    tc_range = c(251, 320),
    m_range = c(0.1, 0.9),
    omega_range = c(6, 13),
    n_starts = 30
  )

  expect_false(is.null(fit))
  expect_equal(fit$tc, tc, tolerance = 10)
  expect_equal(fit$m, m, tolerance = 0.15)
  expect_equal(fit$omega, omega, tolerance = 2)
})

test_that("lppls_filter rejects out-of-range parameters", {
  fit_ok <- list(tc = 280, m = 0.5, omega = 8, converged = TRUE)
  expect_true(lppls_filter(fit_ok))

  fit_bad_m <- list(tc = 280, m = 1.5, omega = 8, converged = TRUE)
  expect_false(lppls_filter(fit_bad_m))

  expect_false(lppls_filter(NULL))
})
