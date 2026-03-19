# test_pdf.R -- Tests for PDF/CDF computation

library(testthat)

root <- rprojroot::find_root(rprojroot::is_rstudio_project)
source(file.path(root, "R", "lppls_pdf.R"))

test_that("lppls_tc_pdf returns valid density", {
  set.seed(42)
  tc_values <- rnorm(100, mean = 300, sd = 10)
  pdf_dt <- lppls_tc_pdf(tc_values)

  expect_true(is.data.table(pdf_dt))
  expect_true(all(c("tc", "density") %in% names(pdf_dt)))
  expect_true(all(pdf_dt$density >= 0))

  # Approximate integration should be close to 1
  integral <- sum(diff(pdf_dt$tc) * pdf_dt$density[-1])
  expect_equal(integral, 1, tolerance = 0.05)
})

test_that("lppls_tc_cdf returns valid cumulative distribution", {
  tc_values <- c(100, 200, 300, 400, 500)
  cdf_dt <- lppls_tc_cdf(tc_values)

  expect_true(is.data.table(cdf_dt))
  expect_equal(nrow(cdf_dt), 5)
  expect_equal(cdf_dt$cdf[5], 1.0)
  expect_true(all(diff(cdf_dt$cdf) >= 0))
})

test_that("lppls_tc_pdf handles single value", {
  pdf_dt <- lppls_tc_pdf(300)
  expect_equal(nrow(pdf_dt), 1)
})

test_that("lppls_tc_cdf handles empty input", {
  cdf_dt <- lppls_tc_cdf(numeric(0))
  expect_equal(nrow(cdf_dt), 0)
})
