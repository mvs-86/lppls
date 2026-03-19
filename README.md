# Deep LPPLS: Neural Network Calibration of the LPPLS Model in R

R implementation of the **Mono-LPPLS-NN (M-LNN)** and **Poly-LPPLS-NN (P-LNN)** models from [Nielsen, Sornette & Raissi (2024)](https://arxiv.org/abs/2405.12803) for estimating the nonlinear parameters of the Log-Periodic Power Law Singularity (LPPLS) model. Includes the classical Levenberg-Marquardt calibration, synthetic data generation, probability density functions (PDFs) of predicted critical times across calibration windows, and publication-quality plots reproducing the paper's figures.

All neural network components use [`torch` for R](https://torch.mlverse.org/) (libtorch C++ bindings) — **no Python dependency**.

## Description

The LPPLS model captures super-exponential growth decorated by log-periodic oscillations preceding critical transitions (financial crashes, material failure, rockslides, etc.). Its formula is:

```
O(t) = A + B(tc - t)^m + C1(tc - t)^m cos(w ln(tc - t)) + C2(tc - t)^m sin(w ln(tc - t))
```

This project provides three calibration methods:

| Method | Description |
|--------|-------------|
| **LM** | Classical Levenberg-Marquardt nonlinear least squares via `minpack.lm` |
| **M-LNN** | Physics-informed neural network trained per time series; LPPLS structure embedded in the loss function |
| **P-LNN** | Supervised neural network trained on large synthetic datasets; near-instant inference on new series |

## Installation

Requires **R 4.5+**.

```r
# Clone the repository
# git clone <repo-url> && cd lppls

# Restore all dependencies via renv
renv::restore()

# Install libtorch binaries (one-time, ~200 MB download)
torch::install_torch()
```

### Dependencies

Core: `data.table`, `ggplot2`, `patchwork`, `torch`, `minpack.lm`, `future`, `future.apply`, `quantmod`, `testthat`, `skimr`, `rprojroot`.

All managed by `renv` — `renv::restore()` handles everything.

## Quickstart

```r
library(data.table)
source("R/lppls_core.R")
source("R/lppls_synthetic.R")

# Generate a synthetic LPPLS series with known parameters
set.seed(42)
params <- generate_lppls_params(1, t_len = 252)
series <- generate_lppls_series(params[1], t_len = 252)
noisy  <- add_white_noise(series$value, amplitude = 0.05)

# Calibrate with Levenberg-Marquardt
fit <- lppls_fit_lm(series$t, noisy, tc_range = c(253, 310))
cat(sprintf("True tc=%.1f  Estimated tc=%.1f\n", params$tc[1], fit$tc))
cat(sprintf("True m=%.2f  Estimated m=%.2f\n", params$m[1], fit$m))
```

## Usage

### 1. Train P-LNN models

Trains three P-LNN variants (white noise, AR(1), both) on 10 000 synthetic series each. Produces loss curve plots and sample training data visualizations.

```bash
Rscript R/run_train_plnn.R
```

Output: `output/plnn_*.pt` (saved models), `output/plnn_*_loss.png`, `output/plnn_*_samples.png`

### 2. Synthetic evaluation (Figure 3)

Evaluates LM, M-LNN, and P-LNN on 250 test scenarios and plots CDFs of absolute parameter errors.

```bash
Rscript R/run_synthetic_eval.R
```

Output: `output/error_cdf_panel.png`, `output/cdf_*.png`

### 3. Real-data analysis (Figures 4-6)

Downloads Nasdaq and Silver ETF data via `quantmod`, runs multi-window calibrations, and produces tc PDF overlay plots.

```bash
Rscript R/run_real_data.R
```

Output: `output/tc_pdf_*.png`

### 4. Run tests

```bash
Rscript -e "testthat::test_dir('tests')"
```

## Project Structure

```
lppls/
├── R/
│   ├── lppls_core.R          # LPPLS formula, basis functions, LM calibration
│   ├── lppls_synthetic.R     # Synthetic data generation (white/AR1/both noise)
│   ├── m_lnn.R               # M-LNN model (torch): physics-informed per-series NN
│   ├── p_lnn.R               # P-LNN model (torch): supervised NN on synthetic data
│   ├── lppls_pdf.R           # Multi-window calibration and tc PDF/CDF computation
│   ├── lppls_plots.R         # Plotting: error CDFs, tc PDF overlays, loss curves
│   ├── utils.R               # Project root helper
│   ├── run_train_plnn.R      # Script: train P-LNN models
│   ├── run_synthetic_eval.R  # Script: evaluate methods on synthetic data
│   └── run_real_data.R       # Script: real-data analysis with quantmod
├── tests/
│   ├── test_lppls_core.R     # Core LPPLS math and LM calibration tests
│   ├── test_synthetic.R      # Synthetic data generation tests
│   ├── test_m_lnn.R          # M-LNN model tests
│   ├── test_p_lnn.R          # P-LNN model tests
│   └── test_pdf.R            # PDF/CDF computation tests
├── output/                   # Generated plots, models, and processed data
├── renv/                     # renv library (auto-managed)
├── renv.lock                 # Locked dependency versions
├── CLAUDE.md                 # Development guidelines
└── README.md
```

## Citation

### Paper

```bibtex
@article{nielsen2024deep,
  title   = {Deep LPPLS: Forecasting of temporal critical points in natural,
             engineering and financial systems},
  author  = {Nielsen, Joshua and Sornette, Didier and Raissi, Maziar},
  journal = {arXiv preprint arXiv:2405.12803},
  year    = {2024}
}
```

### R Packages

| Package | Reference |
|---------|-----------|
| `torch` | Falbel D, Luraschi J (2025). *torch: Tensors and Neural Networks with 'LibTorch' for R*. R package version 0.16.3, https://torch.mlverse.org/ |
| `data.table` | Barrett T, Dowle M, Srinivasan A (2025). *data.table: Extension of 'data.frame'*. R package version 1.18.2, https://r-datatable.com |
| `ggplot2` | Wickham H (2016). *ggplot2: Elegant Graphics for Data Analysis*. Springer-Verlag New York. ISBN 978-3-319-24277-4 |
| `patchwork` | Pedersen TL (2024). *patchwork: The Composer of Plots*. R package version 1.3.2, https://patchwork.data-imaginist.com |
| `minpack.lm` | Elzhov TV, Mullen KM, Spiess A-N, Bolker B (2023). *minpack.lm: R Interface to the Levenberg-Marquardt Nonlinear Least-Squares Algorithm*. R package version 1.2-4 |
| `future` | Bengtsson H (2025). *future: Unified Parallel and Distributed Processing in R*. R package version 1.70.0, https://future.futureverse.org |
| `future.apply` | Bengtsson H (2025). *future.apply: Apply Function to Elements in Parallel using Futures*. R package version 1.20.2 |
| `quantmod` | Ryan JA, Ulrich JM (2024). *quantmod: Quantitative Financial Modelling Framework*. R package version 0.4.28, https://www.quantmod.com |
| `testthat` | Wickham H (2011). *testthat: Get Started with Testing*. The R Journal, 3(1), 5-10 |
| `skimr` | Waring E, Quinn M, McNamara A, Arino de la Rubia E, Zhu H, Ellis S (2025). *skimr: Compact and Flexible Summaries of Data*. R package version 2.2.2 |

## License

MIT
