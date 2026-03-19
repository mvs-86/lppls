# lppls_plots.R -- Visualization functions for Deep LPPLS analysis
#
# Reproduces paper figures:
#   - Fig 3: CDF of absolute parameter errors
#   - Fig 4-6: Time series + LPPLS fits + tc PDF overlay (dual y-axis)
#   - Fig 7: Synthetic training data samples
#   - Fig 8-10: Training/validation loss curves

library(ggplot2)
library(patchwork)
library(data.table)

# --- Fig 3 style: CDF of parameter errors -----------------------------------

plot_error_cdf <- function(results_dt, param_col, title = "",
                           x_label = "Absolute Error") {
  ggplot(results_dt, aes(x = .data[[param_col]], color = method)) +
    stat_ecdf(linewidth = 0.8) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = title,
      x = x_label,
      y = "Cumulative Probability",
      color = "Method"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

plot_error_cdf_panel <- function(results_dt, title = "CDF of Absolute Parameter Errors") {
  p_tc <- plot_error_cdf(results_dt, "tc_error", expression(t[c] ~ "error"))
  p_m <- plot_error_cdf(results_dt, "m_error", expression(m ~ "error"))
  p_omega <- plot_error_cdf(results_dt, "omega_error", expression(omega ~ "error"))
  p_mse <- plot_error_cdf(results_dt, "mse_error", "MSE")

  (p_tc + p_m + p_omega + p_mse) +
    plot_layout(guides = "collect") +
    plot_annotation(title = title) &
    theme(legend.position = "bottom")
}

# --- Fig 4-6 style: Time series + tc PDF overlay ----------------------------

plot_tc_pdf_overlay <- function(obs_dt, pdf_list, t1_first, t2_last,
                                tc_actual = NULL, title = "",
                                date_col = NULL) {
  # obs_dt must have columns: t, value (and optionally date)
  # pdf_list: named list of data.tables with columns tc, density
  has_dates <- !is.null(date_col) && date_col %in% names(obs_dt)

  if (has_dates) {
    obs_dt <- copy(obs_dt)
    setnames(obs_dt, date_col, "date_plot", skip_absent = TRUE)
  }

  # Find PDF y-range for secondary axis scaling
  max_density <- max(sapply(pdf_list, function(d) max(d$density, na.rm = TRUE)))
  obs_range <- range(obs_dt$value, na.rm = TRUE)
  scale_factor <- diff(obs_range) / (max_density * 1.2)

  # Base plot: observed time series
  x_var <- if (has_dates) "date_plot" else "t"

  p <- ggplot(obs_dt, aes(x = .data[[x_var]], y = value)) +
    geom_line(linewidth = 0.6, color = "black")

  # Add calibration window shading
  if (has_dates) {
    t1_val <- obs_dt[t == t1_first, date_plot]
    t2_val <- obs_dt[t == t2_last, date_plot]
  } else {
    t1_val <- t1_first
    t2_val <- t2_last
  }

  p <- p + annotate("rect",
    xmin = t1_val, xmax = t2_val,
    ymin = -Inf, ymax = Inf,
    alpha = 0.15, fill = "grey70"
  )

  # Overlay PDFs as filled areas (mapped to secondary y-axis)
  colors <- c(
    "Levenberg-Marquardt" = "#377eb8",
    "Mono-LPPLS-NN" = "#ff7f00",
    "Poly-LPPLS-NN 100K" = "#984ea3"
  )
  model_names <- names(pdf_list)
  if (is.null(model_names)) model_names <- paste("Model", seq_along(pdf_list))

  for (i in seq_along(pdf_list)) {
    pdf_dt <- copy(pdf_list[[i]])
    # Map tc to x-axis (dates or numeric)
    if (has_dates) {
      # tc is numeric index; map to dates
      pdf_dt[, x_val := obs_dt$date_plot[pmin(pmax(round(tc), 1), nrow(obs_dt))]]
    } else {
      pdf_dt[, x_val := tc]
    }
    # Scale density to fit on primary y-axis
    pdf_dt[, y_scaled := density * scale_factor + obs_range[1]]

    col <- if (model_names[i] %in% names(colors)) {
      colors[model_names[i]]
    } else {
      scales::hue_pal()(length(pdf_list))[i]
    }

    p <- p + geom_area(
      data = pdf_dt,
      aes(x = x_val, y = y_scaled),
      fill = col, alpha = 0.3, color = col, linewidth = 0.5,
      inherit.aes = FALSE
    )
  }

  # Vertical lines
  p <- p +
    geom_vline(xintercept = t1_val, linetype = "dotdash", color = "green4") +
    geom_vline(xintercept = t2_val, linetype = "dotdash", color = "red3")

  if (!is.null(tc_actual)) {
    if (has_dates) {
      tc_line <- obs_dt$date_plot[pmin(round(tc_actual), nrow(obs_dt))]
    } else {
      tc_line <- tc_actual
    }
    p <- p + geom_vline(xintercept = tc_line, linetype = "dashed", color = "black")
  }

  # Secondary y-axis for density
  p <- p + scale_y_continuous(
    name = obs_dt[, ifelse(has_dates, "Price", "Value")],
    sec.axis = sec_axis(
      ~ (. - obs_range[1]) / scale_factor,
      name = "Density"
    )
  )

  p + labs(title = title, x = if (has_dates) "Date" else "Time") +
    theme_minimal() +
    theme(
      axis.title.y.right = element_text(color = "grey40"),
      axis.text.y.right = element_text(color = "grey40")
    )
}

# --- Single LPPLS fit plot --------------------------------------------------

plot_lppls_fit <- function(obs_dt, fit, t1, t2, title = "") {
  fit_dt <- data.table(
    t = obs_dt[t >= t1 & t <= t2, t],
    fitted = fit$fitted_values
  )

  ggplot() +
    geom_line(data = obs_dt, aes(x = t, y = value),
              color = "black", linewidth = 0.5) +
    geom_line(data = fit_dt, aes(x = t, y = fitted),
              color = "red", linewidth = 0.8) +
    geom_vline(xintercept = t1, linetype = "dotdash", color = "green4") +
    geom_vline(xintercept = t2, linetype = "dotdash", color = "red3") +
    geom_vline(xintercept = fit$tc, linetype = "dashed", color = "blue") +
    annotate("rect", xmin = t1, xmax = t2, ymin = -Inf, ymax = Inf,
             alpha = 0.1, fill = "grey") +
    labs(title = title, x = "Time", y = "Value") +
    theme_minimal()
}

# --- Fig 7 style: Synthetic training data samples ---------------------------

plot_synthetic_samples <- function(X_matrix, params_dt = NULL, n_show = 8,
                                   title = "Synthetic LPPLS Training Samples") {
  n_show <- min(n_show, nrow(X_matrix))
  idx <- sample(nrow(X_matrix), n_show)

  plots <- lapply(seq_along(idx), function(k) {
    i <- idx[k]
    dt <- data.table(t = seq_len(ncol(X_matrix)), value = X_matrix[i, ])
    subtitle <- if (!is.null(params_dt)) {
      sprintf("tc=%.1f, m=%.2f, w=%.1f",
              params_dt$tc[i], params_dt$m[i], params_dt$omega[i])
    } else {
      paste("Sample", i)
    }
    ggplot(dt, aes(x = t, y = value)) +
      geom_line(color = "steelblue", linewidth = 0.4) +
      labs(title = subtitle, x = NULL, y = NULL) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 8),
        axis.text = element_text(size = 6)
      )
  })

  wrap_plots(plots, ncol = 4) +
    plot_annotation(title = title)
}

# --- Fig 8-10 style: Training/validation loss curves ------------------------

plot_training_loss <- function(train_loss, val_loss, title = "Training Loss") {
  dt <- data.table(
    epoch = rep(seq_along(train_loss), 2),
    loss = c(train_loss, val_loss),
    type = rep(c("Training Loss", "Validation Loss"), each = length(train_loss))
  )

  ggplot(dt, aes(x = epoch, y = loss, color = type)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = c("Training Loss" = "#377eb8",
                                  "Validation Loss" = "#e41a1c")) +
    labs(title = title, x = "Epoch", y = "Loss", color = NULL) +
    theme_minimal() +
    theme(legend.position = "top")
}
