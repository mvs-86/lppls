# lppls_sf.R -- LPPLS Static Fitting (SF) comparator
#
# Implements the static fitting approach from Ma & Li (2024) Section 2.1, Eq. (3):
#   ln|E[ln(P_t)] - A| = alpha*ln|tc-t| + ln|B| + C1'*cos(omega*ln(tc-t))
#                         + C2'*sin(omega*ln(tc-t))
#
# Given fixed {tc, omega, A}, solves for {alpha, ln|B|, C1', C2'} via OLS.
# Outer optimization over {tc, omega, A} minimizes quantile loss.

library(data.table)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_core.R"))

# --- Design matrix for static fitting ----------------------------------------

#' Build the static fitting design matrix (Appendix A.1).
#'
#' Given fixed tc, omega, A, constructs the N×4 matrix for the linearised
#' log-equation.
#'
#' @param t Numeric vector of observation times.
#' @param obs Numeric vector of log-price observations.
#' @param tc Scalar critical time (must satisfy tc > max(t)).
#' @param omega Scalar angular frequency.
#' @param A Scalar parameter (log-price at critical time).
#' @return N×4 design matrix or NULL if computation fails.
lppls_sf_basis <- function(t, obs, tc, omega, A) {
  dt <- tc - t
  if (any(dt <= 0)) return(NULL)

  lhs <- obs - A
  valid <- abs(lhs) > 1e-15
  if (sum(valid) < 5) return(NULL)

  log_abs_lhs <- log(abs(lhs[valid]))
  log_dt <- log(dt[valid])
  cos_term <- cos(omega * log_dt)
  sin_term <- sin(omega * log_dt)

  list(
    y = log_abs_lhs,
    X = cbind(log_dt, 1, cos_term, sin_term),
    idx = valid
  )
}

# --- Linear solver for static fitting ----------------------------------------

#' Solve for linear parameters in the SF formulation.
#'
#' @param t,obs,tc,omega,A As in \code{lppls_sf_basis}.
#' @return Named vector c(alpha, ln_abs_B, C1_dot, C2_dot) or NAs.
lppls_sf_solve_linear <- function(t, obs, tc, omega, A) {
  parts <- lppls_sf_basis(t, obs, tc, omega, A)
  if (is.null(parts)) return(rep(NA_real_, 4))

  beta <- tryCatch(
    solve(crossprod(parts$X), crossprod(parts$X, parts$y)),
    error = function(e) rep(NA_real_, 4)
  )
  names(beta) <- c("alpha", "ln_abs_B", "C1_dot", "C2_dot")
  beta
}

# --- Quantile loss ------------------------------------------------------------

#' Compute quantile regression loss for SF fit.
#'
#' Uses the check loss (pinball loss) across multiple quantiles of the
#' residuals from the linearised equation.
#'
#' @param t,obs,tc,omega,A Model parameters.
#' @param quantiles Numeric vector of quantile levels.
#' @return Scalar loss value. Returns Inf on failure.
lppls_sf_quantile_loss <- function(t, obs, tc, omega, A,
                                   quantiles = seq(0.1, 0.9, 0.1)) {
  parts <- lppls_sf_basis(t, obs, tc, omega, A)
  if (is.null(parts)) return(Inf)

  beta <- tryCatch(
    solve(crossprod(parts$X), crossprod(parts$X, parts$y)),
    error = function(e) NULL
  )
  if (is.null(beta)) return(Inf)

  resid <- parts$y - parts$X %*% beta
  alpha_est <- beta[1]

  # Filter: alpha should be in valid range

  if (alpha_est < 0.01 || alpha_est > 0.99) return(Inf)

  # Sum of quantile (check) losses
  total_loss <- 0
  for (tau in quantiles) {
    total_loss <- total_loss + sum(resid * (tau - (resid < 0)))
  }
  total_loss / length(quantiles)
}

# --- Multi-start optimizer ----------------------------------------------------

#' Fit LPPLS via static fitting with multi-start L-BFGS-B.
#'
#' Optimizes over {tc, omega, A} to minimize quantile loss. Linear parameters
#' are solved analytically at each evaluation.
#'
#' @param t Numeric vector of observation times.
#' @param obs Numeric vector of log-price observations.
#' @param tc_range Length-2 bounds for tc. Default: [t_end-0.1*dt, t_end+0.2*dt].
#' @param omega_range Length-2 bounds for omega (default c(4.9, 15)).
#' @param A_range Length-2 bounds for A. Default: based on obs range.
#' @param n_starts Number of random starting points (default 50).
#' @param max_iter Maximum iterations per start (default 200).
#' @return list(tc, omega, A, alpha, ln_abs_B, C1_dot, C2_dot, loss) or NULL.
lppls_sf_fit <- function(t, obs,
                         tc_range = NULL,
                         omega_range = c(4.9, 15),
                         A_range = NULL,
                         n_starts = 50L,
                         max_iter = 200L) {
  t2 <- max(t)
  dt_span <- t2 - min(t)

  if (is.null(tc_range)) {
    tc_range <- c(t2 - 0.1 * dt_span, t2 + 0.2 * dt_span)
    tc_range[1] <- max(tc_range[1], t2 + 1)
  }

  if (is.null(A_range)) {
    obs_range <- range(obs)
    A_range <- c(obs_range[1] - diff(obs_range), obs_range[2] + diff(obs_range))
  }

  lower <- c(tc_range[1], omega_range[1], A_range[1])
  upper <- c(tc_range[2], omega_range[2], A_range[2])

  obj_fn <- function(par) {
    lppls_sf_quantile_loss(t, obs, tc = par[1], omega = par[2], A = par[3])
  }

  best <- list(loss = Inf)
  set.seed(42)

  for (i in seq_len(n_starts)) {
    start <- c(
      runif(1, lower[1], upper[1]),
      runif(1, lower[2], upper[2]),
      runif(1, lower[3], upper[3])
    )

    fit <- tryCatch(
      optim(start, obj_fn, method = "L-BFGS-B",
            lower = lower, upper = upper,
            control = list(maxit = max_iter)),
      error = function(e) NULL
    )

    if (!is.null(fit) && fit$value < best$loss) {
      best <- list(
        tc = fit$par[1], omega = fit$par[2], A = fit$par[3],
        loss = fit$value, convergence = fit$convergence
      )
    }
  }

  if (is.infinite(best$loss)) return(NULL)

  lin <- lppls_sf_solve_linear(t, obs, best$tc, best$omega, best$A)
  if (anyNA(lin)) return(NULL)

  list(
    tc = best$tc,
    omega = best$omega,
    A = best$A,
    alpha = unname(lin["alpha"]),
    ln_abs_B = unname(lin["ln_abs_B"]),
    C1_dot = unname(lin["C1_dot"]),
    C2_dot = unname(lin["C2_dot"]),
    loss = best$loss,
    convergence = best$convergence
  )
}
