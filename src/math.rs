//! duckdb_meta — classical meta-analysis (single column, no moderator).
//!
//! Faithful port of metafor::rma.uni (v5.2.1) single-column estimators, which is
//! exactly what JASP's "Classical Meta Analysis" calls (a thin wrapper over
//! metafor::rma). JASP exposes 14 tau² estimators; GENQ/GENQM need
//! CompQuadForm's Farebrother approximation and are deferred, so we port the
//! other 12: EE, FE, ML, REML, DL, HE, HS, HSk, SJ, EB, PM, PMM.
//!
//! Single-column math (X = intercept, p = 1), reduced to scalar sums. With
//! weights wi = 1/(vi + tau2):
//!   Swx  = Σ wi            beta = Swxy / Swx
//!   Swxy = Σ wi·yi         RSS(t) = yᵀP(t)y = Swx2 − Swxy²/Sw   (P symmetric idempotent)
//!   Swx2 = Σ wi·yi²        trP  = Swx − Swx²/k
//!                                trPP = Σ wi² − (Σwi)²/k
//!                                trPV = Σ vi·wi − (Σ vi·wi)²/Sw
//!   Q = RSS(0)  (inverse-variance FE; the heterogeneity test uses FE weights for ALL methods)
//!
//! Estimators (method → tau2) follow rma.uni() verbatim:
//!   HS/HSk/HE/DL closed form; SJ single closed form; EB/ML/REML Fisher scoring
//!   (adj = Swx2−Swxy²/Swx − const); PM/PMM bisection on RSS(tau2) = target.
//! Control defaults (mfopt): tau2.min=0, tau2.max=max(100, 10·mad(y)²),
//! threshold=1e-5, maxiter=100, stepadj=1.

// ── scalar stats (self-contained, no external crates) ───────────────────────

/// Standard normal CDF (erf via Abramowitz & Stegun 26.2.17, |err| < 7.5e-8).
pub fn z_cdf(x: f64) -> f64 {
    let s = if x >= 0.0 { 1.0 } else { -1.0 };
    let a = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t)
            * (-a * a).exp();
    0.5 * (1.0 + s * y)
}

/// Inverse standard-normal CDF (Acklam, |err| < 1.15e-9).
pub fn qnorm(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let p = p.clamp(1e-16, 1.0 - 1e-16);
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q)
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// log-gamma (Lanczos, g=5, |err| < 1e-14 for z > 0.5).
fn lgamma(z: f64) -> f64 {
    const G: [f64; 8] = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        // reflection: ln Γ(z) = ln π − ln sin(πz) − ln Γ(1−z)
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - lgamma(1.0 - z);
    }
    let x = z - 1.0;
    let mut a = 0.99999999999980993;
    for i in 0..G.len() {
        a += G[i] / (x + 1.0 + i as f64);
    }
    let t = x + 7.5;
    0.9189385332046727 + (x + 0.5) * t.ln() - t + a.ln()
}

/// regularized lower incomplete gamma P(a, x), series (x < a+1).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x).exp() * x.powf(a) * (-lgamma(a)).exp()
}

/// regularized upper incomplete gamma Q(a, x), continued fraction (x >= a+1).
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1e300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = if an * d + b < 1e-300 { 1e-300 } else { an * d + b };
        c = if b + an / c < 1e-300 { 1e-300 } else { b + an / c };
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x).exp() * x.powf(a) * (-lgamma(a)).exp() * h
}

/// chi-squared CDF: F(x; df) = P(df/2, x/2).
fn chisq_cdf(x: f64, df: f64) -> f64 {
    let a = df / 2.0;
    let x2 = x / 2.0;
    if x2 <= 0.0 {
        return 0.0;
    }
    if x2 < a + 1.0 {
        gamma_p_series(a, x2)
    } else {
        1.0 - gamma_q_cf(a, x2)
    }
}

/// qchisq(p, df) by bisection.
fn qchisq(p: f64, df: f64) -> f64 {
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    while chisq_cdf(hi, df) < p {
        hi *= 2.0;
        if hi > 1e7 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if chisq_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ── Student-t distribution (for Knapp–Hartung / t test) ────────────────────

/// Regularized upper-tail of the t CDF: P(T > x), T ~ t(df).
/// t CDF is expressed via the regularized incomplete beta function:
/// for x >= 0, 2*P(T > x) = I_{df/(df+x^2)}(df/2, 1/2).
fn t_sf2(x: f64, df: f64) -> f64 {
    let a = df / 2.0;
    let b = 0.5;
    let z = df / (df + x * x);
    ibeta(z, a, b)
}

/// Two-sided t p-value: 2*P(T > |x|).
pub fn t_two_sided_p(x: f64, df: f64) -> f64 {
    if !(df > 0.0) {
        return f64::NAN;
    }
    let x = x.abs();
    if x == 0.0 {
        return 1.0;
    }
    let p = t_sf2(x, df);
    p.clamp(0.0, 1.0)
}

/// t critical value for a two-sided confidence level `level` (i.e. R's
/// qt(level/2, df, lower.tail = FALSE)). Returns q with P(T > q) = (1-level)/2.
pub fn qt(level: f64, df: f64) -> f64 {
    // t_sf2(x,df) = 2*P(T>x); we want 2*P(T>q) = 1-level.
    let target = 1.0 - level;
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    while t_sf2(hi, df) > target {
        hi *= 2.0;
        if hi > 1e7 {
            break;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_sf2(mid, df) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// R's `pt(x, df)`: CDF of the Student-t distribution, P(T <= x).
/// For x >= 0, P(T > x) = 0.5 * I_{df/(df+x^2)}(df/2, 1/2);
/// the lower tail is then 1 minus that, and the symmetry pt(x)+pt(-x)=1 handles x<0.
pub fn pt_tv(x: f64, df: f64) -> f64 {
    if !(df > 0.0) {
        return f64::NAN;
    }
    if !x.is_finite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    if x == 0.0 {
        return 0.5;
    }
    let a = df / 2.0;
    let b = 0.5;
    let z = df / (df + x * x);
    if x > 0.0 {
        (1.0 - 0.5 * ibeta(z, a, b)).clamp(0.0, 1.0)
    } else {
        (0.5 * ibeta(z, a, b)).clamp(0.0, 1.0)
    }
}

/// Regularized incomplete beta function I_x(a, b) via continued fractions
/// (Numerical Recipes `betai`). a, b > 0, 0 <= x <= 1.
fn ibeta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // front factor for the continued-fraction branch:
    // front(x; a, b) = x^a (1-x)^b / (a * B(a,b)).
    let front = |x: f64, a: f64, b: f64| {
        let lb = lgamma(a) + lgamma(b) - lgamma(a + b);
        (a * x.ln() + b * (1.0 - x).ln() - lb).exp() / a
    };
    // Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) to pick the faster branch.
    if x > (a + 1.0) / (a + b + 2.0) {
        1.0 - betacf(1.0 - x, b, a) * front(1.0 - x, b, a)
    } else {
        betacf(x, a, b) * front(x, a, b)
    }
}

/// NR `betacf(a, b, x)`: continued fraction for the regularized incomplete beta.
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAXIT: i32 = 300;
    const EPS: f64 = 3.0e-15;
    const FPM: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPM {
        d = FPM;
    }
    d = 1.0 / d;
    let mut h = d;
    let mut m = 1;
    while m <= MAXIT {
        let m2 = 2 * m;
        // even step
        let aa = (m as f64) * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPM {
            d = FPM;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPM {
            c = FPM;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -((a + m as f64) * (qab + m as f64) * x) / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPM {
            d = FPM;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPM {
            c = FPM;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if del.abs() - 1.0 < EPS {
            break;
        }
        m += 1;
    }
    h
}

/// R's mad(y) (median of |y − median(y)|, constant = 1).
fn mad(y: &[f64]) -> f64 {
    let mut s: Vec<f64> = y.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let med = if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    };
    let mut ad: Vec<f64> = y.iter().map(|&v| (v - med).abs()).collect();
    ad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        ad[n / 2]
    } else {
        0.5 * (ad[n / 2 - 1] + ad[n / 2])
    }
}

// ── estimator core ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct MetaResult {
    pub method: &'static str,
    pub test: &'static str,
    pub k: i64,
    pub df: i64,
    pub beta: f64,
    pub se: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub z: f64,
    pub p: f64,
    pub tau2: f64,
    pub se_tau2: f64,
    pub q: f64,
    pub i2: f64,
    pub h2: f64,
}

/// Weighted sums for one-way (intercept-only) model with weights w (w_i = 1/(vi+t)).
/// X = column of ones; P = W − W·X(X'WX)⁻¹X'W with P_ij = w_i·δ_ij − w_i·w_j/S.
struct Wsum {
    s: f64,     // Σ w
    sw2: f64,   // Σ w²
    sw3: f64,   // Σ w³
    swxy: f64,  // Σ w·y
    swx2: f64,  // Σ w·y²
}

impl Wsum {
    fn compute(w: &[f64], y: &[f64]) -> Self {
        let mut s = 0.0;
        let mut sw2 = 0.0;
        let mut sw3 = 0.0;
        let mut swxy = 0.0;
        let mut swx2 = 0.0;
        for i in 0..w.len() {
            let wi = w[i];
            s += wi;
            sw2 += wi * wi;
            sw3 += wi * wi * wi;
            swxy += wi * y[i];
            swx2 += wi * y[i] * y[i];
        }
        Wsum { s, sw2, sw3, swxy, swx2 }
    }
    // RSS(t) = Y'PY = swx2 − swxy²/S  (P annihilates the intercept)
    fn rss(&self) -> f64 {
        (self.swx2 - self.swxy * self.swxy / self.s).max(0.0)
    }
    // tr(P) = S − Σw²/S
    fn trp(&self) -> f64 {
        self.s - self.sw2 / self.s
    }
    // Σ_ij P_ij² = Σw² − 2Σw³/S + (Σw²)²/S²
    fn sumpp(&self) -> f64 {
        self.sw2 - 2.0 * self.sw3 / self.s + self.sw2 * self.sw2 / (self.s * self.s)
    }
}

/// (trPV, Σ PV·PVᵀ elementwise, Σ PV·P elementwise) for V = diag(v), one-way P.
fn pv_products(w: &[f64], v: &[f64]) -> (f64, f64, f64) {
    let n = w.len();
    let s: f64 = w.iter().sum();
    let mut trpv = 0.0;
    let mut sumpvtpv = 0.0;
    let mut sumpvp = 0.0;
    for i in 0..n {
        for j in 0..n {
            let pij = w[i] * if i == j { 1.0 } else { 0.0 } - w[i] * w[j] / s;
            let pvij = pij * v[j];
            let pvji = pij * v[i];
            if i == j {
                trpv += pij * v[i];
            }
            sumpvtpv += pvij * pvji;
            sumpvp += pvij * pij;
        }
    }
    (trpv, sumpvtpv, sumpvp)
}

fn w_of(v: &[f64], tau2: f64) -> Vec<f64> {
    v.iter().map(|&vi| 1.0 / (vi + tau2)).collect()
}

/// Fisher scoring for ML(0) / REML(1) / EB(2), starting at `start` (HE estimate).
fn fisher(y: &[f64], v: &[f64], kind: i32, start: f64, maxiter: i32) -> f64 {
    let n = y.len() as f64;
    let p = 1.0;
    let threshold = 1e-5;
    let mut t = start.max(0.0);
    for _ in 0..maxiter.max(1) {
        let w = w_of(v, t);
        let ws = Wsum::compute(&w, y);
        // Y'P²Y = Σ w_i² (y_i − β̂)²
        let b = ws.swxy / ws.s;
        let pp_y: f64 = w.iter().zip(y).map(|(&wi, &yi)| {
            let d = yi - b;
            wi * wi * d * d
        }).sum();
        let adj = match kind {
            0 => (pp_y - ws.s) / ws.sw2,                // ML
            1 => (pp_y - ws.trp()) / ws.sumpp(),        // REML
            _ => (ws.rss() * n / (n - p) - n) / ws.s,   // EB
        };
        let mut adj = if adj.is_nan() || !adj.is_finite() { 0.0 } else { adj };
        while t + adj < 0.0 {
            adj = 0.5 * adj;
        }
        let change = (t - (t + adj)).abs();
        t += adj;
        if change < threshold {
            break;
        }
    }
    t
}

/// REML/ML local-max check (ll0check): set tau2=0 if the likelihood is clearly larger at tau2=0.
fn ll0_check(y: &[f64], v: &[f64], kind: i32, tau2: &mut f64) {
    let n = y.len() as f64;
    let p = 1.0;
    let tol = f64::EPSILON.sqrt();
    if *tau2 <= 1e-5 {
        return;
    }
    let l_at = |t: f64| -> f64 {
        let w = w_of(v, t);
        let ws = Wsum::compute(&w, y);
        let rss = ws.rss();
        let logdet_v = v.iter().map(|&vi| (vi + t).ln()).sum::<f64>();
        if kind == 0 {
            -0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * logdet_v - 0.5 * rss
        } else {
            -0.5 * (n - p) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * logdet_v
                - 0.5 * ws.s.ln()
                - 0.5 * rss
        }
    };
    if l_at(0.0) - l_at(*tau2) > tol {
        *tau2 = 0.0;
    }
}

/// PM (target = k−p) / PMM (target = qchisq(.5, k−p)); RSS(t) is monotone decreasing in t.
fn root_pm(y: &[f64], v: &[f64], target: f64, tau2_max: f64) -> f64 {
    let rss_at = |t: f64| {
        let w = w_of(v, t);
        Wsum::compute(&w, y).rss()
    };
    if rss_at(0.0) < target {
        return 0.0;
    }
    if rss_at(tau2_max) > target {
        return tau2_max;
    }
    let mut lo = 0.0f64;
    let mut hi = tau2_max;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if rss_at(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// HE estimate (also the ML/REML/EB start): unweighted P0 = I − (1/k)·11'.
fn he_estimate(y: &[f64], v: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let rss0 = y.iter().map(|&yi| (yi - mean) * (yi - mean)).sum::<f64>();
    let trpv0 = v.iter().sum::<f64>() * (n - 1.0) / n;
    ((rss0 - trpv0) / (n - 1.0)).max(0.0)
}

/// Estimate tau2 for a metafor method code (one-way, no moderator).
fn estimate_tau2(y: &[f64], v: &[f64], method: &str, maxiter: i32) -> f64 {
    let n = y.len() as f64;
    let p = 1.0;
    // FE sums (w = 1/vi)
    let wfe = w_of(v, 0.0);
    let f = Wsum::compute(&wfe, y);
    let q_fe = f.rss();
    let s_fe = f.s;
    let trp_fe = f.trp();
    let he = he_estimate(y, v);
    let tau2_max = 100.0f64.max(10.0 * mad(y).powi(2));
    match method {
        "EE" | "FE" => 0.0,
        "HS" => ((q_fe - n) / s_fe).max(0.0),
        "HSk" => ((n / (n - p) * q_fe - n) / s_fe).max(0.0),
        "DL" => ((q_fe - (n - p)) / trp_fe).max(0.0),
        "HE" => he,
        "SJ" => {
            let mean = y.iter().sum::<f64>() / n;
            let t0 = y.iter().map(|&yi| (yi - mean) * (yi - mean)).sum::<f64>() / n;
            let w = w_of(v, t0);
            let r = Wsum::compute(&w, y).rss();
            (t0 * r / (n - p)).max(0.0)
        }
        "ML" => {
            let mut t = fisher(y, v, 0, he, maxiter);
            ll0_check(y, v, 0, &mut t);
            t
        }
        "REML" => {
            let mut t = fisher(y, v, 1, he, maxiter);
            ll0_check(y, v, 1, &mut t);
            t
        }
        "EB" => fisher(y, v, 2, he, maxiter),
        "PM" => root_pm(y, v, n - p, tau2_max),
        "PMM" => root_pm(y, v, qchisq(0.5, n - p), tau2_max),
        _ => f64::NAN,
    }
}

/// Standard error of tau2 (metafor formulas; NaN for FE/EE).
fn se_tau2(y: &[f64], v: &[f64], method: &str, tau2: f64) -> f64 {
    let n = y.len() as f64;
    let p = 1.0;
    if method == "EE" || method == "FE" {
        return f64::NAN;
    }
    let t2 = tau2.max(0.0);
    // FE weights (w = 1/vi) for HS/HSk/DL
    let wfe = w_of(v, 0.0);
    let f = Wsum::compute(&wfe, y);
    match method {
        "HS" | "HSk" => {
            let base =
                (2.0 * (n - p) + 4.0 * t2 * f.trp() + 2.0 * t2 * t2 * f.sumpp()) / (f.s * f.s);
            let se = base.max(0.0).sqrt();
            if method == "HSk" {
                n / (n - p) * se
            } else {
                se
            }
        }
        "DL" => {
            let base = 2.0 * (n - p) + 4.0 * t2 * f.trp() + 2.0 * t2 * t2 * f.sumpp();
            base.max(0.0).sqrt() / f.trp()
        }
        "HE" => {
            let swv: f64 = v.iter().sum();
            let swv2: f64 = v.iter().map(|&x| x * x).sum();
            // unweighted P0: PV = P0·V; Σ(PV·PVᵀ) elementwise = Σ_i Σ_j P0_ij²·v_i·v_j
            // = (1−2/n)·Σv² + (Σv)²/n²
            let sumpvtpv = (1.0 - 2.0 / n) * swv2 + (swv * swv) / (n * n);
            let trpv0 = swv * (n - 1.0) / n;
            let base = 2.0 * sumpvtpv + 4.0 * t2 * trpv0 + 2.0 * t2 * t2 * (n - p);
            base.max(0.0).sqrt() / (n - p)
        }
        "SJ" => {
            let mean = y.iter().sum::<f64>() / n;
            let t0 = y.iter().map(|&yi| (yi - mean) * (yi - mean)).sum::<f64>() / n;
            let w = w_of(v, t0);
            let (_, sumpvtpv, sumpvp) = pv_products(&w, v);
            let sumpp = {
                let s: f64 = w.iter().sum();
                let sw2: f64 = w.iter().map(|&x| x * x).sum();
                let sw3: f64 = w.iter().map(|&x| x * x * x).sum();
                sw2 - 2.0 * sw3 / s + sw2 * sw2 / (s * s)
            };
            let base = 2.0 * sumpvtpv + 4.0 * t2 * sumpvp + 2.0 * t2 * t2 * sumpp;
            t0 / (n - p) * base.max(0.0).sqrt()
        }
        "ML" => {
            let w = w_of(v, tau2);
            let sw2: f64 = w.iter().map(|&x| x * x).sum();
            (2.0 / sw2).max(0.0).sqrt()
        }
        "REML" => {
            let w = w_of(v, tau2);
            let ws = Wsum::compute(&w, y);
            (2.0 / ws.sumpp()).max(0.0).sqrt()
        }
        "EB" | "PM" | "PMM" => {
            let w = w_of(v, tau2);
            let s: f64 = w.iter().sum();
            (2.0 * n * n / (n - p) / (s * s)).max(0.0).sqrt()
        }
        _ => f64::NAN,
    }
}

/// Single-column meta-analysis: pooled estimate + heterogeneity.
pub fn meta_pool(
    y: &[f64],
    v: &[f64],
    method: &str,
    level: f64,
    maxiter: i32,
    test: &str,
) -> Option<MetaResult> {
    // test = "z" (Wald) or "Knapp-Hartung" (t, JASP default).
    let is_kh = !test.eq_ignore_ascii_case("z");
    // Drop studies with missing/invalid values (metafor's na.action = "na.omit").
    let yv: Vec<(f64, f64)> = y
        .iter()
        .zip(v)
        .filter(|(&yi, &vi)| yi.is_finite() && vi.is_finite() && vi > 0.0)
        .map(|(&yi, &vi)| (yi, vi))
        .collect();
    if yv.len() < 2 {
        return None;
    }
    let yy: Vec<f64> = yv.iter().map(|t| t.0).collect();
    let vv: Vec<f64> = yv.iter().map(|t| t.1).collect();
    let mcode = normalize_method(method)?;
    let n = yy.len() as f64;
    let p = 1.0;
    let zc = qnorm(1.0 - (1.0 - level) / 2.0);
    let is_fe = mcode == "EE" || mcode == "FE";
    let tau2 = estimate_tau2(&yy, &vv, mcode, maxiter);
    // pooled estimate & SE from inverse-variance weights at the final tau2
    let w = w_of(&vv, tau2);
    let ws = Wsum::compute(&w, &yy);
    let beta = ws.swxy / ws.s;
    let se0 = (1.0 / ws.s).sqrt();
    let st2 = se_tau2(&yy, &vv, mcode, tau2);

    // Knapp–Hartung: scale the covariance by s2w = RSS.knha/(k−p), where
    // RSS.knha uses the SAME (random-effects) inverse-variance weights as the
    // estimate; then test beta against a t(k−p). (metafor rma.uni, test="knha".)
    let df = (n - p) as i64;
    let rss_kh = ws.rss();
    let (se, crit, pval, stat) = if is_kh {
        let s2w = if rss_kh <= f64::EPSILON {
            0.0
        } else {
            rss_kh / (n - p)
        };
        let se_kh = (s2w / ws.s).sqrt();
        let tcrit = qt(1.0 - (1.0 - level) / 2.0, df as f64);
        let tv = beta / se_kh;
        (se_kh, tcrit, 2.0 * pt_tv(-tv.abs(), df as f64), tv)
    } else {
        let tv = beta / se0;
        (se0, zc, 2.0 * (1.0 - z_cdf(tv.abs())), tv)
    };

    // heterogeneity (always FE inverse-variance weights)
    let wfe = w_of(&vv, 0.0);
    let f = Wsum::compute(&wfe, &yy);
    let q = f.rss();
    let trp_fe = f.trp();
    let vt = (n - p) / trp_fe; // "typical" sampling variance (i2def = "1")
    let i2 = if is_fe {
        if q > 0.0 {
            ((q - (n - p)) / q).max(0.0) * 100.0
        } else {
            0.0
        }
    } else {
        100.0 * tau2 / (vt + tau2)
    };
    let h2 = if is_fe {
        q / (n - p)
    } else {
        tau2 / vt + 1.0
    };
    Some(MetaResult {
        method: mcode,
        test: if is_kh { "Knapp-Hartung" } else { "z" },
        k: n as i64,
        df,
        beta,
        se,
        ci_lo: beta - crit * se,
        ci_hi: beta + crit * se,
        z: stat,
        p: pval,
        tau2,
        se_tau2: st2,
        q,
        i2,
        h2,
    })
}

fn normalize_method(s: &str) -> Option<&'static str> {
    match s.trim().to_ascii_uppercase().as_str() {
        "EE" | "EQUAL" | "EFFECTS" => Some("EE"),
        "FE" | "FIXED" => Some("FE"),
        "ML" => Some("ML"),
        "REML" | "RMLE" => Some("REML"),
        "DL" => Some("DL"),
        "HE" | "HEDGES" => Some("HE"),
        "HS" => Some("HS"),
        "HSK" => Some("HSk"),
        "SJ" => Some("SJ"),
        "EB" => Some("EB"),
        "PM" | "PAULE" => Some("PM"),
        "PMM" => Some("PMM"),
        _ => None,
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        (a - b).abs() / (1.0 + a.abs()).max(b.abs()) <= 1e-5
    }

    fn lines(path: &str) -> Vec<Vec<String>> {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        let raw = std::fs::read_to_string(full).unwrap_or_else(|e| panic!("read {path}: {e}"));
        raw.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split(',')
                    .map(|f| f.trim().trim_matches('"').to_string())
                    .collect()
            })
            .collect()
    }

    /// datasets from meta_datasets.csv: name -> (y, v) in file order; NA -> f64::NAN.
    fn load_datasets() -> Vec<(String, Vec<f64>, Vec<f64>)> {
        let rows = lines("test/fixtures/meta_datasets.csv");
        let num = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN);
        let mut map: std::collections::HashMap<String, (Vec<f64>, Vec<f64>)> = Default::default();
        let mut order: Vec<String> = Vec::new();
        for r in rows.iter().skip(1) {
            let name = r[0].as_str().to_string();
            if !order.contains(&name) {
                order.push(name.clone());
            }
            let e = map.entry(name.clone()).or_default();
            e.0.push(num(&r[1]));
            e.1.push(num(&r[2]));
        }
        order
            .into_iter()
            .map(|n| (n.clone(), map[&n].0.clone(), map[&n].1.clone()))
            .collect()
    }

    #[test]
    fn primitives_match() {
        // z_cdf vs known values
        assert!((z_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((z_cdf(1.959963985) - 0.975).abs() < 1e-6);
        assert!((z_cdf(-1.959963985) - 0.025).abs() < 1e-6);
        // qnorm round-trips
        assert!((qnorm(0.975) - 1.959963985).abs() < 1e-6);
        assert!((qnorm(0.5) - 0.0).abs() < 1e-9);
        // qchisq: median of chi2(1) ≈ 0.4549
        assert!((qchisq(0.5, 1.0) - 0.4549364).abs() < 1e-4);
        // chisq_cdf: P(Q < 3.841, df=1) ≈ 0.95
        assert!((chisq_cdf(3.8414588, 1.0) - 0.95).abs() < 1e-5);
        // lgamma sanity: lgamma(5) = ln(24)
        assert!((lgamma(5.0) - 24.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn t_dist_matches_r() {
        // R 4.5.2 reference values
        assert!((pt_tv(-2.0, 5.0) - 0.050969739414929).abs() < 1e-6);
        assert!((pt_tv(-1.959963985, 50.0) - 0.027792602057372).abs() < 1e-6);
        assert!((pt_tv(-2.570581728, 10.0) - 0.013931424868052).abs() < 1e-6);
        assert!((qt(0.975, 5.0) - 2.57058183563631).abs() < 1e-6);
        assert!((qt(0.995, 10.0) - 3.16927267261695).abs() < 1e-6);
        assert!((qt(0.95, 3.0) - 2.35336343480182).abs() < 1e-6);
        assert!((qt(0.025, 7.0) + 2.36462425159278).abs() < 1e-6);
        // symmetry: pt(-x) + pt(x) = 1
        assert!(((pt_tv(-1.0, 4.0) + pt_tv(1.0, 4.0)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn meta_pool_matches_metafor() {
        let rows = lines("test/fixtures/meta_ref.csv");
        let hdr = rows[0].clone();
        let idx = |name: &str| hdr.iter().position(|h| h == name).unwrap();
        let datasets = load_datasets();
        let mut checked = 0;
        let mut failed = 0;
        for r in rows.iter().skip(1) {
            let name = &r[idx("name")];
            let method = &r[idx("method")];
            let d = datasets
                .iter()
                .find(|d| d.0 == *name)
                .unwrap_or_else(|| panic!("dataset {name} missing"));
            let (y, v) = (d.1.as_slice(), d.2.as_slice());
            let want_beta = r[idx("beta")].parse::<f64>().unwrap();
            let want_se = r[idx("se")].parse::<f64>().unwrap();
            let want_tau2 = r[idx("tau2")].parse::<f64>().unwrap();
            let want_se_tau2 = r[idx("se_tau2")].parse::<f64>().unwrap_or(f64::NAN);
            let want_q = r[idx("Q")].parse::<f64>().unwrap();
            let want_i2 = r[idx("I2")].parse::<f64>().unwrap();
            let want_h2 = r[idx("H2")].parse::<f64>().unwrap();
            let got = meta_pool(y, v, method, 0.95, 100, "z").expect("pool failed");
            let mut errs: Vec<String> = Vec::new();
            if !close(got.beta, want_beta) { errs.push(format!("beta {} vs {}", got.beta, want_beta)); }
            if !close(got.se, want_se) { errs.push(format!("se {} vs {}", got.se, want_se)); }
            if !close(got.tau2, want_tau2) { errs.push(format!("tau2 {} vs {}", got.tau2, want_tau2)); }
            if !close(got.q, want_q) { errs.push(format!("q {} vs {}", got.q, want_q)); }
            if !close(got.i2, want_i2) { errs.push(format!("i2 {} vs {}", got.i2, want_i2)); }
            if !close(got.h2, want_h2) { errs.push(format!("h2 {} vs {}", got.h2, want_h2)); }
            if want_se_tau2.is_finite() && !close(got.se_tau2, want_se_tau2) {
                errs.push(format!("se_tau2 {} vs {}", got.se_tau2, want_se_tau2));
            }
            // Knapp-Hartung t-test (JASP default) vs metafor rma(test="knha").
            let kh = meta_pool(y, v, method, 0.95, 100, "Knapp-Hartung").expect("kh failed");
            let w_kh_t = r[idx("kh_t")].parse::<f64>().unwrap();
            let w_kh_p = r[idx("kh_p")].parse::<f64>().unwrap();
            let w_kh_se = r[idx("kh_se")].parse::<f64>().unwrap();
            let w_kh_lo = r[idx("kh_CIlo")].parse::<f64>().unwrap();
            let w_kh_hi = r[idx("kh_CIhi")].parse::<f64>().unwrap();
            let w_df = r[idx("df")].parse::<i64>().unwrap();
            if !close(kh.z, w_kh_t) { errs.push(format!("kh_t {} vs {}", kh.z, w_kh_t)); }
            if !close(kh.p, w_kh_p) { errs.push(format!("kh_p {} vs {}", kh.p, w_kh_p)); }
            if !close(kh.se, w_kh_se) { errs.push(format!("kh_se {} vs {}", kh.se, w_kh_se)); }
            if !close(kh.ci_lo, w_kh_lo) { errs.push(format!("kh_lo {} vs {}", kh.ci_lo, w_kh_lo)); }
            if !close(kh.ci_hi, w_kh_hi) { errs.push(format!("kh_hi {} vs {}", kh.ci_hi, w_kh_hi)); }
            if kh.df != w_df { errs.push(format!("df {} vs {}", kh.df, w_df)); }
            checked += 1;
            if !errs.is_empty() {
                failed += 1;
                eprintln!("FAIL {name}/{method}: {}", errs.join("; "));
            }
        }
        assert_eq!(failed, 0, "{failed}/{checked} rows failed");
    }
}
