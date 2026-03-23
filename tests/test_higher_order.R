# test_higher_order.R -- Tests for 2nd and 3rd order LPPLS

library(testthat)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_higher_order.R"))
source(file.path(root, "R", "lppls_core.R"))

test_that("lppls_basis_order2 produces correct dimensions", {
  t <- seq(0, 99)
  X <- lppls_basis_order2(t, tc = 110, alpha = 0.5, omega = 8,
                          delta_t = 50, delta_omega = -5)
  expect_equal(dim(X), c(100, 4))
  expect_true(all(is.finite(X)))
})

test_that("lppls_basis_order3 produces correct dimensions", {
  t <- seq(0, 99)
  X <- lppls_basis_order3(t, tc = 110, alpha = 0.5, omega = 8,
                          delta_t = 50, delta_omega = -5,
                          delta_t_prime = 60, delta_omega_prime = -3)
  expect_equal(dim(X), c(100, 4))
  expect_true(all(is.finite(X)))
})

test_that("order 2 reduces to order 1 when delta_t -> Inf", {
  t <- seq(0, 99)
  tc <- 110; alpha <- 0.5; omega <- 8

  X1 <- lppls_basis(t, tc, alpha, omega)
  X2 <- lppls_basis_order2(t, tc, alpha, omega,
                           delta_t = 1e12, delta_omega = 0)

  expect_equal(X2, X1, tolerance = 1e-6)
})

test_that("lppls_value_order2 returns correct length", {
  t <- seq(0, 99)
  vals <- lppls_value_order2(t, tc = 110, alpha = 0.5, omega = 8,
                             delta_t = 50, delta_omega = -5,
                             A = 7, B = -0.5, C1 = 0.02, C2 = -0.01)
  expect_length(vals, 100)
  expect_true(all(is.finite(vals)))
})

test_that("lppls_value_order3 returns correct length", {
  t <- seq(0, 99)
  vals <- lppls_value_order3(t, tc = 110, alpha = 0.5, omega = 8,
                             delta_t = 50, delta_omega = -5,
                             delta_t_prime = 60, delta_omega_prime = -3,
                             A = 7, B = -0.5, C1 = 0.02, C2 = -0.01)
  expect_length(vals, 100)
  expect_true(all(is.finite(vals)))
})

test_that("linear solver recovers exact params for order 2", {
  t <- seq(0, 99)
  tc <- 110; alpha <- 0.5; omega <- 8
  delta_t <- 50; delta_omega <- -5
  A <- 7; B <- -0.5; C1 <- 0.02; C2 <- -0.01

  obs <- lppls_value_order2(t, tc, alpha, omega, delta_t, delta_omega,
                            A, B, C1, C2)
  lin <- lppls_solve_linear_order2(t, obs, tc, alpha, omega,
                                   delta_t, delta_omega)

  expect_equal(unname(lin), c(A, B, C1, C2), tolerance = 1e-8)
})

test_that("numerical stability with edge-case parameters", {
  t <- seq(0, 99)
  # alpha near 0
  X <- lppls_basis_order2(t, tc = 110, alpha = 0.01, omega = 5,
                          delta_t = 50, delta_omega = -1)
  expect_true(all(is.finite(X)))

  # alpha near 1
  X <- lppls_basis_order2(t, tc = 110, alpha = 0.99, omega = 15,
                          delta_t = 50, delta_omega = -1)
  expect_true(all(is.finite(X)))
})
