
suppressMessages(library(metafor))
y <- c(0.82,-0.15,0.45,1.10,0.30,-0.60,0.72,0.05)
v <- c(0.090,0.049,0.064,0.025,0.036,0.058,0.030,0.081)
for (m in c("REML","FE")) {
  f <- rma(yi=y, vi=v, method=m, test="knha")
  cat("=== ", m, " knha ===\n")
  cat("se     ", format(f$se, digits=15), "\n")
  cat("zval   ", format(f$zval, digits=15), "\n")
  cat("pval   ", format(f$pval, digits=15), "\n")
  cat("ci.lb  ", format(f$ci.lb, digits=15), "\n")
  cat("ci.ub  ", format(f$ci.ub, digits=15), "\n")
  cat("ddf    ", f$ddf, "\n")
  fz <- rma(yi=y, vi=v, method=m, test="z")
  cat("-- z --\n  se     ", format(fz$se, digits=15), "\n")
  cat("  zval   ", format(fz$zval, digits=15), "\n")
  cat("  pval   ", format(fz$pval, digits=15), "\n")
  cat("  ci.lb  ", format(fz$ci.lb, digits=15), "\n")
  cat("  ci.ub  ", format(fz$ci.ub, digits=15), "\n")
  cat("  ddf    ", fz$ddf, "\n")
}
# verify KH by hand for REML
fr <- rma(yi=y, vi=v, method="REML")
t2 <- fr$tau2
w <- 1/(v+t2)
S <- sum(w)
beta <- sum(w*y)/S
se0 <- 1/sqrt(S)
rss <- sum(w*(y-beta)^2)
s2w <- rss/length(y)
sekh <- se0*sqrt(s2w)
cat("manual REML kh: se0", format(se0,digits=12), "s2w", format(s2w,digits=12), "sekh", format(sekh,digits=12), "stat", format(beta/sekh,digits=12), "\n")
