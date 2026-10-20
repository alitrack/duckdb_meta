# Oracle fixtures for duckdb_meta from metafor (v5.2.1). JASP's meta-analysis is
# a thin wrapper over metafor::rma (method default REML, test default Knapp-Hartung).
# This ports the same 12 single-column tau^2 estimators (JASP exposes 14;
# GENQ/GENQM need CompQuadForm's Farebrother and are deferred) plus z-test,
# Knapp-Hartung t-test and forest weights.
# Run: Rscript gen_meta_ref.R  (writes test/fixtures/meta_ref.csv + meta_forest_ref.csv)
suppressPackageStartupMessages(library(metafor))

METHODS <- c("EE", "FE", "ML", "REML", "DL", "HE", "HS", "HSk", "SJ", "EB", "PM", "PMM")

# Knapp-Hartung t-test given tau2 (0 for fixed-effect). Mirrors metafor rma.uni
# test="knha": w = 1/(vi+tau2), beta = W'Y/W, vb = 1/W, s2w = RSS.knha/(k-p),
# se = sqrt(s2w*vb), t = beta/se ~ t(k-p). RSS.knha uses the SAME (RE) weights.
kh <- function(yi, vi, tau2, level = 0.95) {
  k <- length(yi); p <- 1L
  w <- 1 / (vi + tau2)
  beta <- sum(w * yi) / sum(w); vb <- 1 / sum(w)
  rss <- sum(w * (yi - beta)^2)
  s2w <- if (rss <= .Machine$double.eps) 0 else rss / (k - p)
  se <- sqrt(s2w * vb)
  tcrit <- qt(1 - (1 - level) / 2, k - p)
  t <- beta / se
  list(tau2 = tau2, beta = beta, se = se, t = t, df = k - p,
       p = 2 * pt(-abs(t), k - p), CIlo = beta - tcrit * se, CIhi = beta + tcrit * se)
}

fit_one <- function(yi, vi, method, name) {
  k <- length(yi); p <- 1L
  qz <- qnorm(0.975)
  is_fe <- method %in% c("FE", "EE")
  if (is_fe) {
    w <- 1 / vi
    beta <- sum(w * yi) / sum(w); se <- sqrt(1 / sum(w))
    Q <- sum(w * (yi - beta)^2)
    z <- beta / se; pval <- 2 * pnorm(-abs(z))
    I2 <- if (Q > 0) max(0, (Q - (k - p)) / Q) * 100 else 0
    H2 <- if (k > p) Q / (k - p) else NA
    kx <- kh(yi, vi, 0)
    return(data.frame(name = name, method = method, beta = beta, se = se,
      CIlo = beta - qz * se, CIhi = beta + qz * se, z = z, p = pval,
      tau2 = 0, se_tau2 = NA, Q = Q, I2 = I2, H2 = H2, k = k,
      kh_t = kx$t, kh_p = kx$p, kh_se = kx$se, kh_CIlo = kx$CIlo, kh_CIhi = kx$CIhi,
      df = kx$df, stringsAsFactors = FALSE))
  }
  fit <- rma.uni(yi = yi, vi = vi, method = method,
    control = if (method %in% c("PM", "PMM")) list(tol = 1e-12) else NULL)
  beta <- fit$beta[1]; se <- fit$se[1]
  z <- beta / se; pval <- 2 * pnorm(-abs(z))
  kx <- kh(yi, vi, fit$tau2)
  data.frame(
    name = name, method = method,
    beta = beta, se = se,
    CIlo = beta - qz * se, CIhi = beta + qz * se,
    z = z, p = pval,
    tau2 = fit$tau2, se_tau2 = fit$se.tau2,
    Q = fit$QE, I2 = fit$I2, H2 = fit$H2,
    k = k,
    kh_t = kx$t, kh_p = kx$p, kh_se = kx$se, kh_CIlo = kx$CIlo, kh_CIhi = kx$CIhi,
    df = kx$df, stringsAsFactors = FALSE
  )
}

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
    v6 <- 1 / n6; v6[13] <- NA  # one NULL se -> study dropped, k=24
    list(name = "d6_forest",
       y = 0.6 + rnorm(k6, 0, sqrt(0.2)) + rnorm(k6, 0, 1/sqrt(n6)), v = v6) }
)

rows <- list()
for (d in datasets) {
  keep <- !is.na(d$v) & !is.na(d$y)
  for (m in METHODS) rows[[length(rows) + 1]] <- fit_one(d$y[keep], d$v[keep], m, d$name)
}
ref <- do.call(rbind, rows)
write.csv(ref, "fixtures/meta_ref.csv", row.names = FALSE)

# forest fixture for d6 (FE + REML study weights & CIs)
{
  d <- datasets[[6]]
  keep <- !is.na(d$v) & !is.na(d$y)
  yi <- d$y[keep]; vi <- d$v[keep]
  qz <- qnorm(0.975)
  wfe <- 1 / vi; bfe <- sum(wfe * yi) / sum(wfe)
  fr <- rma.uni(yi, vi, method = "REML"); t2r <- fr$tau2
  wr <- 1 / (vi + t2r)
  f <- data.frame(
    y = yi, v = vi,
    ci_fe_lo = yi - qz * sqrt(vi), ci_fe_hi = yi + qz * sqrt(vi), w_fe = wfe / sum(wfe),
    ci_re_lo = yi - qz * sqrt(vi + t2r), ci_re_hi = yi + qz * sqrt(vi + t2r), w_re = wr / sum(wr)
  )
  write.csv(f, "fixtures/meta_forest_ref.csv", row.names = FALSE)
}

cat("wrote", nrow(ref), "rows\n")
print(head(ref, 14))
