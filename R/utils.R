# utils.R -- Project utility functions

project_root <- function() {
  rprojroot::find_root(rprojroot::is_rstudio_project)
}

source_r <- function(file) {
  source(file.path(project_root(), "R", file))
}
