# lppls_core.R -- LPPLS model mathematics and Levenberg-Marquardt calibration
#
# LPPLS formula (Eq. 4-5 from Nielsen et al. 2024):
#   O(t) = A + B*f(t) + C1*g(t) + C2*h(t)
# where:
#   f(t) = (tc - t)^m
#   g(t) = (tc - t)^m * cos(omega * ln(tc - t))
#   h(t) = (tc - t)^m * sin(omega * ln(tc - t))

library(data.table)

# --- Core LPPLS functions ---------------------------------------------------

#' Build the LPPLS design matrix for the linear sub-problem.
#'
#' Computes the four basis columns \code{[1, f, g, h]} where
#' \code{f = (tc-t)^m}, \code{g = f * cos(omega * ln(tc-t))},
#' \code{h = f * sin(omega * ln(tc-t))}.
#'
#' @param t Numeric vector of observation times. All values must satisfy
#'   \code{t < tc}.
#' @param tc Scalar critical time.
#' @param m Scalar exponent (typically 0 < m < 1).
#' @param omega Scalar log-periodic angular frequency.
#' @return An \code{N x 4} matrix with columns \code{[1, f, g, h]}.
lppls_basis <- function(t, tc, m, omega) {
  dt <- tc - t
  stopifnot(all(dt > 0))
  dt_m <- dt^m
  log_dt <- log(dt)
  f <- dt_m
  g <- dt_m * cos(omega * log_dt)
  h <- dt_m * sin(omega * log_dt)
  cbind(1, f, g, h)
}

#' Evaluate the LPPLS model at given time points.
#'
#' Computes \code{O(t) = A + B*f + C1*g + C2*h} (Eq. 4-5 from Nielsen et al.
#' 2024).
#'
#' @param t Numeric vector of observation times (\code{t < tc}).
#' @param tc Scalar critical time.
#' @param m Scalar exponent.
#' @param omega Scalar log-periodic angular frequency.
#' @param A,B,C1,C2 Scalar linear parameters.
#' @return Numeric vector of LPPLS values, same length as \code{t}.
lppls_value <- function(t, tc, m, omega, A, B, C1, C2) {
  X <- lppls_basis(t, tc, m, omega)
  as.numeric(X %*% c(A, B, C1, C2))
}

#' Solve for the four linear LPPLS parameters analytically.
#'
#' Given fixed nonlinear parameters \code{(tc, m, omega)}, solves the normal
#' equations \code{(X'X) beta = X' obs} for \code{{A, B, C1, C2}}.
#'
#' @param t Numeric vector of observation times.
#' @param obs Numeric vector of observed values (same length as \code{t}).
#' @param tc,m,omega Scalar nonlinear parameters.
#' @return Named numeric vector \code{c(A, B, C1, C2)}. Returns \code{NA}s if
#'   the system is singular.
lppls_solve_linear <- function(t, obs, tc, m, omega) {
  X <- lppls_basis(t, tc, m, omega)
  beta <- tryCatch(
    solve(crossprod(X), crossprod(X, obs)),
    error = function(e) rep(NA_real_, 4)
  )
  names(beta) <- c("A", "B", "C1", "C2")
  beta
}

#' Compute LPPLS residuals for given nonlinear parameters.
#'
#' Solves the linear parameters internally via \code{\link{lppls_solve_linear}}
#' and returns \code{obs - fitted}.
#'
#' @inheritParams lppls_solve_linear
#' @return Numeric vector of residuals. Returns a large-valued vector
#'   (\code{1e6}) if the linear system is singular.
lppls_residuals <- function(t, obs, tc, m, omega) {
  lin <- lppls_solve_linear(t, obs, tc, m, omega)
  if (anyNA(lin)) return(rep(1e6, length(obs)))
  obs - lppls_value(t, tc, m, omega, lin["A"], lin["B"], lin["C1"], lin["C2"])
}

#' Sum of squared LPPLS residuals.
#'
#' @inheritParams lppls_solve_linear
#' @return Scalar sum of squared errors.
lppls_sse <- function(t, obs, tc, m, omega) {
  sum(lppls_residuals(t, obs, tc, m, omega)^2)
}

# --- Levenberg-Marquardt calibration ----------------------------------------

#' Calibrate the LPPLS model via multi-start Levenberg-Marquardt.
#'
#' Runs \code{n_starts} random initialisations of \code{(tc, m, omega)} within
#' the specified bounds and keeps the solution with the lowest SSE. Linear
#' parameters are solved analytically at each evaluation. Uses
#' \code{minpack.lm::nls.lm}.
#'
#' @param t Numeric vector of observation times.
#' @param obs Numeric vector of observed values.
#' @param tc_range Length-2 numeric vector \code{c(lower, upper)} for the
#'   critical time. Defaults to \code{[max(t)+1, max(t) + 0.2*range(t)]}.
#' @param m_range Length-2 numeric bounds for the exponent (default
#'   \code{c(0.1, 0.9)}).
#' @param omega_range Length-2 numeric bounds for the angular frequency
#'   (default \code{c(6, 13)}).
#' @param n_starts Integer number of random starting points (default 25).
#' @return A list with components \code{tc, m, omega, A, B, C1, C2, sse,
#'   converged, fitted_values}, or \code{NULL} if no run converged.
lppls_fit_lm <- function(t, obs,
                         tc_range = NULL,
                         m_range = c(0.1, 0.9),
                         omega_range = c(6, 13),
                         n_starts = 25) {
  if (is.null(tc_range)) {
    t2 <- max(t)
    tc_range <- c(t2 + 1, t2 + 0.2 * (t2 - min(t)))
  }

  resid_fn <- function(par, t, obs) {
    tc <- par[1]; m <- par[2]; omega <- par[3]
    if (tc <= max(t) || m <= 0 || m >= 1) return(rep(1e6, length(obs)))
    tryCatch(
      lppls_residuals(t, obs, tc, m, omega),
      error = function(e) rep(1e6, length(obs))
    )
  }

  best <- list(sse = Inf)
  set.seed(42)

  for (i in seq_len(n_starts)) {
    tc0 <- runif(1, tc_range[1], tc_range[2])
    m0 <- runif(1, m_range[1], m_range[2])
    omega0 <- runif(1, omega_range[1], omega_range[2])

    fit <- tryCatch(
      minpack.lm::nls.lm(
        par = c(tc = tc0, m = m0, omega = omega0),
        fn = resid_fn,
        lower = c(tc_range[1], m_range[1], omega_range[1]),
        upper = c(tc_range[2], m_range[2], omega_range[2]),
        t = t, obs = obs,
        control = minpack.lm::nls.lm.control(maxiter = 200, nprint = 0)
      ),
      error = function(e) NULL
    )

    if (!is.null(fit)) {
      sse <- sum(fit$fvec^2)
      if (sse < best$sse) {
        best <- list(
          tc = fit$par["tc"], m = fit$par["m"], omega = fit$par["omega"],
          sse = sse, converged = fit$info %in% 1:3
        )
      }
    }
  }

  if (is.infinite(best$sse)) return(NULL)

  lin <- lppls_solve_linear(t, obs, best$tc, best$m, best$omega)
  fitted <- lppls_value(t, best$tc, best$m, best$omega,
                        lin["A"], lin["B"], lin["C1"], lin["C2"])

  list(
    tc = unname(best$tc), m = unname(best$m), omega = unname(best$omega),
    A = lin["A"], B = lin["B"], C1 = lin["C1"], C2 = lin["C2"],
    sse = best$sse, converged = best$converged, fitted_values = fitted
  )
}

# --- Post-fit filter --------------------------------------------------------

#' Filter an LPPLS fit for validity.
#'
#' Checks convergence, parameter bounds, and minimum number of log-periodic
#' oscillations (>= 2.5).
#'
#' @param fit A list returned by \code{\link{lppls_fit_lm}} or an equivalent
#'   structure with components \code{tc, m, omega, converged}.
#' @param m_range Length-2 numeric bounds for \code{m}.
#' @param omega_range Length-2 numeric bounds for \code{omega}.
#' @param max_rel_error Reserved for future use (relative-error filter).
#' @return Logical scalar: \code{TRUE} if the fit passes all checks.
lppls_filter <- function(fit,
                         m_range = c(0.1, 0.9),
                         omega_range = c(6, 13),
                         max_rel_error = 0.05) {
  if (is.null(fit)) return(FALSE)
  if (!fit$converged) return(FALSE)
  if (fit$m < m_range[1] || fit$m > m_range[2]) return(FALSE)
  if (fit$omega < omega_range[1] || fit$omega > omega_range[2]) return(FALSE)
  n_osc <- fit$omega / (2 * pi) * log(fit$tc / 1)
  if (n_osc < 2.5) return(FALSE)
  TRUE
}
