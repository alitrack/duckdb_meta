# emit the raw datasets (after dropping NAs) so Rust can mirror the input
suppressPackageStartupMessages(library(metafor))
datasets <- list(
  list(name = "d1_small",   y = c(0.82, -0.15, 0.45, 1.10, 0.30, -0.60, 0.72, 0.05),
       v = c(0.090, 0.049, 0.064, 0.025, 0.036, 0.058, 0.030, 0.081)),
  { set.seed(42); n2 <- c(8, 15, 22, 5, 30, 12, 18, 9, 25, 14, 6, 20)
    list(name = "d2_unbal",
       y = 0.5 + rnorm(length(n2), 0, sqrt(0.35)) + rnorm(length(n2), 0, 1/sqrt(n2)),
       v = 1 / n2) },
  list(name = "d3_zeroish", y = c(0.12, 0.18, 0.07, 0.22, 0.10, 0.15, 0.09, 0.20, 0.13, 0.17),
       v = c(0.040, 0.025, 0.030, 0.020, 0.035, 0.015, 0.028, 0.018, 0.022, 0.012)),
  list(name = "d4_neg",     y = c(-0.9, -0.4, -1.3, -0.1, -0.6, -1.1, -0.25),
       v = c(0.05, 0.03, 0.04, 0.02, 0.06, 0.035, 0.025)),
  list(name = "d5_extreme", y = c(2.5, -1.8, 0.4, 3.1, -0.2, 1.9, -2.3, 0.8, -1.1),
       v = c(0.12, 0.08, 0.06, 0.10, 0.05, 0.09, 0.07, 0.11, 0.06)),
  { set.seed(7); k6 <- 25; n6 <- sample(4:60, k6)
    v6 <- 1 / n6; v6[13] <- NA
    list(name = "d6_forest",
       y = 0.6 + rnorm(k6, 0, sqrt(0.2)) + rnorm(k6, 0, 1/sqrt(n6)), v = v6) }
)
out <- file("fixtures/meta_datasets.csv", "w")
writeLines("name,y,v", out)
for (d in datasets) {
  keep <- !is.na(d$v) & !is.na(d$y)
  for (i in which(keep)) {
    cat(d$name, ",", formatC(d$y[i], digits=15), ",", formatC(d$v[i], digits=15), "\n", sep="", file=out)
  }
}
close(out)
cat("wrote datasets\n")
