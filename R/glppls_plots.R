# glppls_plots.R -- Visualization functions for G-LPPLS-NN analysis
#
# Reproduces paper figures from Ma & Li (2024):
#   - Fig 1: CDF of simulation errors (4 cols × 3 rows)
#   - Figs 2-4: Bubble detection with 95% PI rectangles + marginal PDFs

library(ggplot2)
library(patchwork)
library(data.table)

# --- Color scheme -------------------------------------------------------------

GLPPLS_COLORS <- c(
  "LPPLS-RF"         = "#377eb8",
  "LPPLS-SF"         = "#a65628",
  "G-LPPLS-NN"       = "#4daf4a",
  "G-LPPLS-NN-Tuned" = "#e41a1c"
)

# --- Bubble detection plot (Figs 2-4 main panel) -----------------------------

#' Plot bubble detection with 95% prediction interval rectangles.
#'
#' @param obs_dt data.table with columns t, value (and optionally date).
#' @param predictions Named list of model predictions. Each element has
#'   mu_tc, mu_A, sigma_tc, sigma_A (or tc/A point estimates).
#' @param tc_actual Scalar actual critical time (optional).
#' @param price_actual Scalar actual critical price (optional).
#' @param title Plot title.
#' @param date_col Name of date column in obs_dt (optional).
#' @return ggplot object.
plot_glppls_bubble <- function(obs_dt, predictions,
                               tc_actual = NULL, price_actual = NULL,
                               title = "", date_col = NULL) {
  has_dates <- !is.null(date_col) && date_col %in% names(obs_dt)
  x_var <- if (has_dates) date_col else "t"

  # Grey out last 10%
  n <- nrow(obs_dt)
  cutoff <- round(0.9 * n)

  p <- ggplot() +
    geom_line(data = obs_dt[seq_len(cutoff)],
              aes(x = .data[[x_var]], y = value),
              color = "black", linewidth = 0.6) +
    geom_line(data = obs_dt[(cutoff + 1):n],
              aes(x = .data[[x_var]], y = value),
              color = "grey60", linewidth = 0.6)

  # Add 95% PI rectangles for each model
  for (model_name in names(predictions)) {
    pred <- predictions[[model_name]]
    col <- if (model_name %in% names(GLPPLS_COLORS)) {
      GLPPLS_COLORS[model_name]
    } else {
      "grey50"
    }

    if (all(c("mu_tc", "sigma_tc", "mu_A", "sigma_A") %in% names(pred))) {
      tc_lo <- pred$mu_tc - 1.96 * pred$sigma_tc
      tc_hi <- pred$mu_tc + 1.96 * pred$sigma_tc
      A_lo <- exp(pred$mu_A - 1.96 * exp(pred$sigma_A))
      A_hi <- exp(pred$mu_A + 1.96 * exp(pred$sigma_A))

      if (has_dates) {
        tc_lo_x <- obs_dt[[date_col]][pmin(pmax(round(tc_lo), 1), n)]
        tc_hi_x <- obs_dt[[date_col]][pmin(pmax(round(tc_hi), 1), n)]
      } else {
        tc_lo_x <- tc_lo
        tc_hi_x <- tc_hi
      }

      p <- p + annotate("rect",
        xmin = tc_lo_x, xmax = tc_hi_x,
        ymin = A_lo, ymax = A_hi,
        alpha = 0.15, fill = col, color = col, linewidth = 0.5
      )
    }
  }

  # Actual critical point
  if (!is.null(tc_actual)) {
    tc_line <- if (has_dates) {
      obs_dt[[date_col]][pmin(round(tc_actual), n)]
    } else {
      tc_actual
    }
    p <- p + geom_vline(xintercept = tc_line, linetype = "dashed",
                        color = "red3", linewidth = 0.7)
  }
  if (!is.null(price_actual)) {
    p <- p + geom_hline(yintercept = price_actual, linetype = "dashed",
                        color = "red3", linewidth = 0.7)
  }

  p + labs(title = title, x = if (has_dates) "Date" else "Time",
           y = "Price") +
    theme_minimal()
}

# --- Full bubble plot with marginal PDFs (Figs 2-4) --------------------------

#' Plot bubble detection with marginal tc and A density panels.
#'
#' @inheritParams plot_glppls_bubble
#' @return patchwork object.
plot_glppls_bubble_full <- function(obs_dt, predictions,
                                    tc_actual = NULL, price_actual = NULL,
                                    title = "", date_col = NULL) {
  # Main panel
  p_main <- plot_glppls_bubble(obs_dt, predictions, tc_actual, price_actual,
                               "", date_col)

  # Top marginal: tc densities
  tc_data <- rbindlist(lapply(names(predictions), function(nm) {
    pred <- predictions[[nm]]
    if (all(c("mu_tc", "sigma_tc") %in% names(pred))) {
      x_seq <- seq(pred$mu_tc - 3 * pred$sigma_tc,
                   pred$mu_tc + 3 * pred$sigma_tc, length.out = 200)
      data.table(tc = x_seq,
                 density = dnorm(x_seq, pred$mu_tc, pred$sigma_tc),
                 model = nm)
    }
  }))

  p_top <- ggplot() + theme_void()
  if (nrow(tc_data) > 0) {
    p_top <- ggplot(tc_data, aes(x = tc, y = density, color = model)) +
      geom_line(linewidth = 0.7) +
      scale_color_manual(values = GLPPLS_COLORS, drop = FALSE) +
      theme_minimal() +
      theme(
        axis.text.x = element_blank(),
        axis.title = element_blank(),
        legend.position = "none",
        plot.margin = margin(0, 0, 0, 0)
      )
    if (!is.null(tc_actual)) {
      p_top <- p_top + geom_vline(xintercept = tc_actual, linetype = "dashed",
                                  color = "red3", linewidth = 0.5)
    }
  }

  # Right marginal: exp(A) densities
  a_data <- rbindlist(lapply(names(predictions), function(nm) {
    pred <- predictions[[nm]]
    if (all(c("mu_A", "sigma_A") %in% names(pred))) {
      exp_sigma <- exp(pred$sigma_A)
      x_seq <- seq(exp(pred$mu_A) - 3 * exp_sigma,
                   exp(pred$mu_A) + 3 * exp_sigma, length.out = 200)
      x_seq <- x_seq[x_seq > 0]
      if (length(x_seq) < 2) return(NULL)
      data.table(price = x_seq,
                 density = dnorm(x_seq, exp(pred$mu_A), exp_sigma),
                 model = nm)
    }
  }))

  p_right <- ggplot() + theme_void()
  if (nrow(a_data) > 0) {
    p_right <- ggplot(a_data, aes(x = density, y = price, color = model)) +
      geom_line(linewidth = 0.7) +
      scale_color_manual(values = GLPPLS_COLORS, drop = FALSE) +
      theme_minimal() +
      theme(
        axis.text.y = element_blank(),
        axis.title = element_blank(),
        legend.position = "none",
        plot.margin = margin(0, 0, 0, 0)
      )
    if (!is.null(price_actual)) {
      p_right <- p_right + geom_hline(yintercept = price_actual,
                                      linetype = "dashed", color = "red3",
                                      linewidth = 0.5)
    }
  }

  # Compose: top-right empty, top-center = tc PDF, left-center = main,
  #          right-center = A PDF
  layout <- "
  BB#
  AAC
  "

  p_main + p_top + p_right +
    plot_layout(design = layout, heights = c(1, 4), widths = c(4, 1)) +
    plot_annotation(title = title) &
    theme(legend.position = "bottom")
}

# --- Simulation CDF panel (Fig 1) --------------------------------------------

#' Plot CDFs of simulation errors in a 4-column × 3-row grid.
#'
#' @param results_dt data.table with columns: order (1/2/3), method,
#'   error_mu_tc, error_sigma_tc, error_mu_A, error_sigma_exp_A.
#' @param title Overall title.
#' @return patchwork object.
plot_glppls_sim_cdf <- function(results_dt, title = "Simulation CDF") {
  error_cols <- c("error_mu_tc", "error_sigma_tc",
                  "error_mu_A", "error_sigma_exp_A")
  col_labels <- c(
    expression("|" * mu[t[c]] ~ "error|"),
    expression("|" * sigma[t[c]] ~ "error|"),
    expression("|" * mu[A] ~ "error|"),
    expression("ln(|" * sigma[e^A] ~ "error| + 1)")
  )
  order_labels <- c("1st Order", "2nd Order", "3rd Order")

  plots <- list()
  for (row in 1:3) {
    for (col in seq_along(error_cols)) {
      dt_sub <- results_dt[order == row]
      ecol <- error_cols[col]

      # Transform 4th column
      if (col == 4) {
        dt_sub <- copy(dt_sub)
        dt_sub[, (ecol) := log(abs(get(ecol)) + 1)]
      }

      p <- ggplot(dt_sub, aes(x = abs(.data[[ecol]]), color = method)) +
        stat_ecdf(linewidth = 0.6) +
        scale_color_manual(values = GLPPLS_COLORS, drop = FALSE) +
        theme_minimal() +
        theme(
          legend.position = "none",
          plot.title = element_text(size = 8),
          axis.text = element_text(size = 6)
        )

      if (row == 1) p <- p + labs(title = col_labels[col])
      if (row == 3) p <- p + labs(x = "Error")
      else p <- p + labs(x = NULL)

      if (col == 1) p <- p + labs(y = order_labels[row])
      else p <- p + labs(y = NULL)

      plots <- c(plots, list(p))
    }
  }

  wrap_plots(plots, ncol = 4, nrow = 3) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title) &
    theme(legend.position = "bottom")
}
