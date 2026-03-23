# lppls_higher_order.R -- 2nd and 3rd order Landau LPPLS expansions
#
# Implements higher-order LPPLS formulas from Ma & Li (2024) Appendix A:
#   - 2nd order (Eq. A.4): adds delta_t, delta_omega parameters
#   - 3rd order (Eq. A.6): adds delta_t_prime, delta_omega_prime parameters
# Both orders share the same 4-column linear structure [1, f, g, h].

library(data.table)

# --- 2nd Order LPPLS (Eq. A.4) -----------------------------------------------

#' Build the 2nd-order LPPLS design matrix.
#'
#' Computes the four basis columns \code{[1, envelope, envelope*cos(phase),
#' envelope*sin(phase)]} using the 2nd-order Landau expansion.
#'
#' @param t Numeric vector of observation times (\code{t < tc}).
#' @param tc Scalar critical time.
#' @param alpha Scalar exponent (0 < alpha < 1).
#' @param omega Scalar log-periodic angular frequency.
#' @param delta_t Scalar 2nd-order time-scale parameter.
#' @param delta_omega Scalar 2nd-order frequency correction.
#' @return An \code{N x 4} matrix with columns \code{[1, f, g, h]}.
lppls_basis_order2 <- function(t, tc, alpha, omega, delta_t, delta_omega) {
  dt <- tc - t
  stopifnot(all(dt > 0))

  dt_ratio <- (dt / delta_t)^(2 * alpha)
  envelope <- dt^alpha / sqrt(1 + dt_ratio)
  phase <- omega * log(dt) + delta_omega / (2 * alpha) * log(1 + dt_ratio)

  f <- envelope
  g <- envelope * cos(phase)
  h <- envelope * sin(phase)
  cbind(1, f, g, h)
}

#' Evaluate the 2nd-order LPPLS model.
#'
#' @inheritParams lppls_basis_order2
#' @param A,B,C1,C2 Scalar linear parameters.
#' @return Numeric vector of model values.
lppls_value_order2 <- function(t, tc, alpha, omega, delta_t, delta_omega,
                               A, B, C1, C2) {
  X <- lppls_basis_order2(t, tc, alpha, omega, delta_t, delta_omega)
  as.numeric(X %*% c(A, B, C1, C2))
}

# --- 3rd Order LPPLS (Eq. A.6) -----------------------------------------------

#' Build the 3rd-order LPPLS design matrix.
#'
#' Extends 2nd-order with additional \code{delta_t_prime} and
#' \code{delta_omega_prime} parameters.
#'
#' @inheritParams lppls_basis_order2
#' @param delta_t_prime Scalar 3rd-order time-scale parameter.
#' @param delta_omega_prime Scalar 3rd-order frequency correction.
#' @return An \code{N x 4} matrix with columns \code{[1, f, g, h]}.
lppls_basis_order3 <- function(t, tc, alpha, omega, delta_t, delta_omega,
                               delta_t_prime, delta_omega_prime) {
  dt <- tc - t
  stopifnot(all(dt > 0))

  dt_ratio <- (dt / delta_t)^(2 * alpha)
  dt_ratio_prime <- (dt / delta_t_prime)^(2 * alpha)

  envelope <- dt^alpha / sqrt((1 + dt_ratio) * (1 + dt_ratio_prime))

  phase <- omega * log(dt) +
    delta_omega / (2 * alpha) * log(1 + dt_ratio) +
    delta_omega_prime / (2 * alpha) * log(1 + dt_ratio_prime)

  f <- envelope
  g <- envelope * cos(phase)
  h <- envelope * sin(phase)
  cbind(1, f, g, h)
}

#' Evaluate the 3rd-order LPPLS model.
#'
#' @inheritParams lppls_basis_order3
#' @param A,B,C1,C2 Scalar linear parameters.
#' @return Numeric vector of model values.
lppls_value_order3 <- function(t, tc, alpha, omega, delta_t, delta_omega,
                               delta_t_prime, delta_omega_prime,
                               A, B, C1, C2) {
  X <- lppls_basis_order3(t, tc, alpha, omega, delta_t, delta_omega,
                          delta_t_prime, delta_omega_prime)
  as.numeric(X %*% c(A, B, C1, C2))
}

# --- Linear solver for higher orders -----------------------------------------

#' Solve linear parameters for 2nd-order LPPLS.
#'
#' @inheritParams lppls_basis_order2
#' @param obs Numeric vector of observed values.
#' @return Named numeric vector \code{c(A, B, C1, C2)}.
lppls_solve_linear_order2 <- function(t, obs, tc, alpha, omega,
                                      delta_t, delta_omega) {
  X <- lppls_basis_order2(t, tc, alpha, omega, delta_t, delta_omega)
  beta <- tryCatch(
    as.numeric(solve(crossprod(X), crossprod(X, obs))),
    error = function(e) rep(NA_real_, 4)
  )
  names(beta) <- c("A", "B", "C1", "C2")
  beta
}

#' Solve linear parameters for 3rd-order LPPLS.
#'
#' @inheritParams lppls_basis_order3
#' @param obs Numeric vector of observed values.
#' @return Named numeric vector \code{c(A, B, C1, C2)}.
lppls_solve_linear_order3 <- function(t, obs, tc, alpha, omega,
                                      delta_t, delta_omega,
                                      delta_t_prime, delta_omega_prime) {
  X <- lppls_basis_order3(t, tc, alpha, omega, delta_t, delta_omega,
                          delta_t_prime, delta_omega_prime)
  beta <- tryCatch(
    as.numeric(solve(crossprod(X), crossprod(X, obs))),
    error = function(e) rep(NA_real_, 4)
  )
  names(beta) <- c("A", "B", "C1", "C2")
  beta
}
