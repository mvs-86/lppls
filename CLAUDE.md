# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

R project for LPPLS (Log-Periodic Power Law Singularity) model implementation/analysis. Uses `renv` for dependency management with R 4.5.

## Environment

- **R version**: 4.5+
- **Primary packages**: `data.table`, `ggplot2`, `patchwork`, `skimr`, `future`, `future.apply`


## Environment Setup

```bash
# Restore renv dependencies (run from R console or Rscript)
Rscript -e "renv::restore()"

# After adding new packages, snapshot the lockfile
Rscript -e "renv::snapshot()"
```

## Code Style

- 2-space indentation (spaces, not tabs)
- UTF-8 encoding
- Strip trailing whitespace; auto-append newline at end of files
- Idiomatic, concise R. No redundant comments or over-explanation.
- Prefer vectorised operations; avoid explicit loops unless unavoidable.
- Use `<-` for assignment. Reserve `=` for function arguments.
- Snake_case for variables and functions. UPPER_SNAKE for constants.
- Max line length: 100 characters.
- Always `set.seed()` when randomness is involved.

## Data Manipulation — data.table first

Default to `data.table` for all data work. Use the `[i, j, by]` paradigm throughout.
```r
dt <- fread("data.csv")

dt[col > 10, .(col1, col2)]                             # filter + select
dt[, new_col := col1 / col2]                            # in-place mutation
dt[, .(mean_val = mean(val, na.rm = TRUE)), by = grp]   # aggregate
dt[col > 0][, .(total = sum(v)), by = g][order(-total)] # chain

# Joins
merge(dt1, dt2, by = "key", all.x = TRUE)

# Reshape
melt(dt, id.vars = "id", variable.name = "metric", value.name = "value")
dcast(dt_long, id ~ metric, value.var = "value")

# Multi-column ops
dt[, lapply(.SD, mean, na.rm = TRUE), by = grp, .SDcols = is.numeric]
```

**Allow alternatives when justified:**
- `dplyr`/`tidyr` if user explicitly requests it
- `stringr` for heavy string work
- `lubridate` for complex date/time logic
- `sf` for geospatial data (pair with ggplot2)

---

## Quick Data Summarization — skimr

Use `skimr::skim()` as the default first-pass summary on any new dataset. Prefer it over `summary()` or `str()` for initial EDA.
```r
library(skimr)

skim(dt)                          # full overview: types, missingness, distributions
skim(dt, col1, col2)              # targeted skim on specific columns
dt[grp == "A"] |> skim()         # skim on a filtered subset

# Capture as data.table for programmatic use
skim_result <- as.data.table(skim(dt))
skim_result[skim_type == "numeric", .(skim_variable, numeric.mean, numeric.sd, n_missing)]
```

- Run `skim()` before any transformation pipeline to understand missingness, types, and ranges.
- Use `skim_without_charts()` in non-interactive / script contexts.

---

## Parallelism — futureverse

Use the `future` ecosystem for all parallel workloads. Never use `parallel::mclapply` or `foreach` directly.
```r
library(future)
library(future.apply)

# Set backend once at the top of the script
plan(multisession, workers = parallelly::availableCores(omit = 1))

# Drop back to sequential (e.g. inside functions/packages)
on.exit(plan(sequential))
```

### Choosing the right apply

| Task | Function |
|---|---|
| List/vector iteration | `future_lapply()` |
| Return vector | `future_sapply()` |
| Side effects only | `future_walk()` (via `furrr`) |
| Over data.frame rows | `future_apply()` |
| Map over multiple inputs | `furrr::future_map2()`, `future_pmap()` |
```r
# Parallelise over a list of datasets
results <- future_lapply(file_list, function(f) {
  dt <- fread(f)
  dt[, .(mean_val = mean(val, na.rm = TRUE)), by = grp]
})
combined <- rbindlist(results)

# furrr for purrr-style mapping
library(furrr)
results <- future_map(param_grid, \(p) fit_model(dt, p), .options = furrr_options(seed = TRUE))
```

### Reproducibility in parallel

Always use `.options = furrr_options(seed = TRUE)` with `furrr`, or `future.seed = TRUE` with `future.apply`:
```r
future_lapply(1:100, function(i) rnorm(10), future.seed = TRUE)
```

### Guidelines

- Set `plan()` once per script at the top level; never inside loops or functions (unless resetting with `on.exit`).
- Prefer `multisession` for cross-platform compatibility; use `multicore` only on Linux/macOS when forking is safe.
- For heavy data.table workloads, benchmark first — `data.table` already uses OpenMP threading internally; adding `future` on top can hurt performance.
- Keep exported objects in futures small; avoid passing large `data.table` objects into workers unnecessarily — use file paths and `fread` inside the future instead.

---

## Visualization — ggplot2 + patchwork

Default theme is `theme_minimal()`. Always set `labs()`. Save with `ggsave()`.
```r
library(ggplot2)
library(patchwork)

ggplot(dt, aes(x = x_var, y = y_var)) +
  geom_point() +
  labs(title = "Title", x = "X", y = "Y") +
  theme_minimal()

ggsave("output/plot.png", width = 8, height = 5, dpi = 150)
```

- **Color**: `scale_color_brewer()` / `scale_fill_brewer()` for grouped data. No rainbow palettes.
- **Interactive plots**: `plotly::ggplotly()` or `echarts4r` when interactivity is requested.

### Multi-panel layouts — patchwork

Use `patchwork` for all multi-panel figures. Avoid `facet_wrap()`/`facet_grid()` when panels show
different variables or geom types; reserve native faceting only for the same geom across factor levels.
```r
p1 <- ggplot(dt, aes(x)) + geom_histogram(bins = 30) + theme_minimal()
p2 <- ggplot(dt, aes(grp, y)) + geom_boxplot() + theme_minimal()
p3 <- ggplot(dt, aes(x, y)) + geom_point() + theme_minimal()

# Compose
p1 + p2                          # side by side
p1 / p2                          # stacked
(p1 + p2) / p3                   # 2-col top, full-width bottom

# Shared title + tag panels
(p1 + p2 + p3) +
  plot_annotation(
    title      = "Dataset Overview",
    tag_levels = "A"
  )

# Align axes across panels
p1 + p2 + plot_layout(axis_titles = "collect")

ggsave("output/panel.png", width = 12, height = 8, dpi = 150)
```

---

## Statistical Analysis

Use base R (`lm`, `glm`, `t.test`, `aov`, etc.). Prepare data with `data.table`. Tidy results with `broom`.
```r
dt_model <- dt[!is.na(outcome)]
fit       <- lm(outcome ~ pred1 + pred2, data = dt_model)
results   <- as.data.table(broom::tidy(fit))
```

---

## Reusable Functions & Packages

- Validate inputs with `stopifnot()` or explicit `stop()`.
- Use explicit defaults on all arguments.
- Namespace all calls in package code (`data.table::fread()`).
- Return invisibly for side-effect functions.
- Document with `roxygen2`; export selectively.
```r
summarise_groups <- function(dt, value_col, group_col) {
  stopifnot(is.data.table(dt), is.character(value_col), is.character(group_col))
  dt[, .(
    mean = mean(get(value_col), na.rm = TRUE),
    sd   = sd(get(value_col),   na.rm = TRUE),
    n    = .N
  ), by = group_col]
}
```

---

## Script Structure
```r
library(data.table)
library(ggplot2)
library(patchwork)
library(skimr)
library(future)
library(future.apply)

plan(multisession, workers = parallelly::availableCores(omit = 1))

# --- Load ---
dt <- fread("data/input.csv")

# --- Summarise ---
skim(dt)

# --- Transform (parallelised over files/chunks if needed) ---
dt[, ratio := x / y]

# --- Analyse ---
fit <- lm(outcome ~ ratio, data = dt)

# --- Visualise ---
p1 <- ggplot(dt, aes(ratio)) + geom_histogram(bins = 30) +
  labs(title = "Distribution", x = "Ratio") + theme_minimal()

p2 <- ggplot(dt, aes(ratio, outcome)) + geom_point() + geom_smooth(method = "lm") +
  labs(title = "Outcome vs Ratio") + theme_minimal()

final_plot <- p1 / p2 + plot_annotation(tag_levels = "A")
ggsave("output/plot.png", final_plot, width = 8, height = 8, dpi = 150)
fwrite(dt, "output/processed.csv")
```

---

## Output Conventions

| Artifact | Path |
|---|---|
| Scripts | `output/<task>.R` |
| Plots | `output/<name>.png` |
| Processed data | `output/<name>_processed.csv` |

Always create the output dir: `dir.create("output", showWarnings = FALSE, recursive = TRUE)`