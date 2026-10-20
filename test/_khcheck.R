suppressPackageStartupMessages(library(metafor))
kh <- function(yi, vi, tau2, level = 0.95) {
  k <- length(yi); p <- 1L
  w <- 1 / (vi + tau2); beta <- sum(w * yi) / sum(w); vb <- 1 / sum(w)
  rss <- sum(w * (yi - beta)^2)
  s2w <- if (rss <= .Machine$double.eps) 0 else rss / (k - p)
  se <- sqrt(s2w * vb); tcrit <- qt(1 - (1 - level) / 2, k - p)
  t <- beta / se
  list(beta = beta, se = se, t = t, p = 2 * pt(-abs(t), k - p),
       CIlo = beta - tcrit * se, CIhi = beta + tcrit * se)
}
y1 <- c(0.82, -0.15, 0.45, 1.10, 0.30, -0.60, 0.72, 0.05)
v1 <- c(0.090, 0.049, 0.064, 0.025, 0.036, 0.058, 0.030, 0.081)
for (m in c("REML", "FE", "DL", "PMM")) {
  f <- rma.uni(y1, v1, method = m, test = "knha",
    control = if (m == "PMM") list(tol = 1e-12) else NULL)
  mytau2 <- if (m == "FE") 0 else f$tau2
  kx <- kh(y1, v1, mytau2)
  cat(sprintf("%-5s beta rma=%-12.6f mine=%-12.6f  se rma=%-12.6f mine=%-12.6f\n",
      m, f$beta, kx$beta, f$se, kx$se))
  cat(sprintf("      t  rma=%-12.6f mine=%-12.6f  p  rma=%-12.6g mine=%-12.6g\n",
      f$zval, kx$t, f$pval, kx$p))
  cat(sprintf("      CI rma=[%-10.6f %-.6f] mine=[%-10.6f %-.6f]\n",
      f$ci.lb, f$ci.ub, kx$CIlo, kx$CIhi))
}
