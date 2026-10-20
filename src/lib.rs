//! duckdb_meta — classical meta-analysis as DuckDB table functions.
//!
//! Faithful port of JASP's Classical Meta Analysis, which is a thin wrapper
//! over `metafor::rma.uni` (single column, no moderator). All math lives in
//! `src/math.rs`, asserted by `cargo test` against R/metafor-generated
//! fixtures (test/gen_meta_ref.R). This file is only the C-API glue.
//!
//! Functions:
//!   meta(y, se [, method, level, maxiter, test])
//!       y  : LIST(DOUBLE) study effect sizes
//!       se : LIST(DOUBLE) standard errors (NULL rows -> study dropped)
//!       method (named, default 'REML'):
//!             EE | FE | ML | REML | DL | HE | HS | HSk | SJ | EB | PM | PMM
//!       level  (named, default 0.95) confidence level
//!       maxiter(named, default 100)  iterations for ML/REML/PM/PMM
//!       test   (named, default 'Knapp-Hartung'):
//!             Knapp-Hartung | z
//!             JASP default is Knapp-Hartung: beta tested against a t(k-1)
//!             using s2w = RSS/(k-1); 'z' is the plain Wald test.
//!       -> one row: method, test, k, df, beta, se, ci_lo, ci_hi, z, p,
//!                    tau2, se_tau2, q, i2, h2
//!          (z is the test statistic: t under Knapp-Hartung, z under 'z';
//!           se / ci are scaled accordingly)
//!
//!   meta_forest(y, se)
//!       -> one row per study (after dropping missing):
//!          study, y, v, ci_fe_lo, ci_fe_hi, w_fe, ci_re_lo, ci_re_hi, w_re
//!       FE weights 1/v, REML weights 1/(v + tau2_REML) (JASP forest defaults).

pub mod math;

use quack_rs::{
    entry_point::init_extension_v2,
    prelude::*,
    table::{BindInfo, TableFunctionBuilder},
    types::{LogicalType, TypeId},
    vector::VectorWriter,
};
use libduckdb_sys::{
    duckdb_data_chunk_get_vector, duckdb_data_chunk_set_size,
    duckdb_extension_access, duckdb_extension_info, duckdb_get_double,
    duckdb_get_list_child, duckdb_get_list_size, idx_t,
};

use math::{meta_pool, qnorm};

// ── LIST(DOUBLE) parameter helpers ─────────────────────────────────
// quack's Value has no list accessor; read via the stable C API.
// duckdb_get_list_child returns a borrowed value (no destroy note in the
// stable C-API docs, unlike duckdb_bind_get_parameter).

/// Read a LIST(DOUBLE) positional parameter into a Vec<f64> (NaN for NULLs).
fn read_list_param(bind: &BindInfo, index: u64) -> Vec<f64> {
    unsafe {
        let raw = bind.get_parameter(index);
        if raw.is_null() {
            return Vec::new();
        }
        let n = duckdb_get_list_size(raw);
        let mut out = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        for i in 0..n {
            let child = duckdb_get_list_child(raw, i);
            // duckdb_get_double returns NULL for a NULL child (stable C API); NaN propagates.
            let v = if child.is_null() { f64::NAN } else { duckdb_get_double(child) };
            out.push(v);
        }
        out
    }
}

// ── meta() ─────────────────────────────────────────────────────────

struct PoolState {
    rows: Option<math::MetaResult>,
}

fn build_meta() -> Result<TableFunctionBuilder, ExtensionError> {
    TableFunctionBuilder::new("meta")
        .param_logical(LogicalType::list(TypeId::Double)) // y
        .param_logical(LogicalType::list(TypeId::Double)) // se
        .named_param("method", TypeId::Varchar)
        .named_param("level", TypeId::Double)
        .named_param("maxiter", TypeId::BigInt)
        .named_param("test", TypeId::Varchar)
        .with_state::<PoolState, _>(|bind| {
            let y = read_list_param(bind, 0);
            let se = read_list_param(bind, 1);
            if y.len() != se.len() {
                return Err(ExtensionError::new(
                    "meta: y and se must have the same length",
                ));
            }
            // meta_pool expects sampling variances v = se^2 (metafor's `vi`)
            let v: Vec<f64> = se.iter().map(|s| s * s).collect();
            let method = unsafe { bind.get_named_parameter_value("method") }.as_str_or("REML");
            let level = unsafe { bind.get_named_parameter_value("level") }.as_f64_or(0.95);
            let maxiter = unsafe { bind.get_named_parameter_value("maxiter") }.as_i64_or(100);
            let test = unsafe { bind.get_named_parameter_value("test") }.as_str_or("Knapp-Hartung");
            if !(0.0 < level && level < 1.0) {
                return Err(ExtensionError::new("meta: level must be in (0, 1)"));
            }
            let result = meta_pool(&y, &v, &method, level, maxiter as i32, &test)
                .ok_or_else(|| {
                    ExtensionError::new(
                        "meta: need at least 2 studies with finite y and positive se",
                    )
                })?;
            bind.add_result_column("method", TypeId::Varchar);
            bind.add_result_column("test", TypeId::Varchar);
            bind.add_result_column("k", TypeId::BigInt);
            bind.add_result_column("df", TypeId::BigInt);
            bind.add_result_column("beta", TypeId::Double);
            bind.add_result_column("se", TypeId::Double);
            bind.add_result_column("ci_lo", TypeId::Double);
            bind.add_result_column("ci_hi", TypeId::Double);
            bind.add_result_column("z", TypeId::Double);
            bind.add_result_column("p", TypeId::Double);
            bind.add_result_column("tau2", TypeId::Double);
            bind.add_result_column("se_tau2", TypeId::Double);
            bind.add_result_column("q", TypeId::Double);
            bind.add_result_column("i2", TypeId::Double);
            bind.add_result_column("h2", TypeId::Double);
            bind.set_cardinality(1, true);
            Ok(PoolState { rows: Some(result) })
        })
        .scan(|state, chunk| {
            let r = state.rows.take();
            if r.is_none() {
                unsafe { duckdb_data_chunk_set_size(chunk.as_raw(), 0) };
                return Ok(());
            }
            let r = r.unwrap();
            unsafe {
                // method
                {
                    let vec = duckdb_data_chunk_get_vector(chunk.as_raw(), 0);
                    let mut w = VectorWriter::new(vec);
                    w.write_str(0, r.method);
                }
                // test
                {
                    let vec = duckdb_data_chunk_get_vector(chunk.as_raw(), 1);
                    let mut w = VectorWriter::new(vec);
                    w.write_str(0, r.test);
                }
                // k
                {
                    let vec = duckdb_data_chunk_get_vector(chunk.as_raw(), 2);
                    let mut w = VectorWriter::new(vec);
                    w.write_i64(0, r.k);
                }
                // df
                {
                    let vec = duckdb_data_chunk_get_vector(chunk.as_raw(), 3);
                    let mut w = VectorWriter::new(vec);
                    w.write_i64(0, r.df);
                }
                // doubles: beta, se, ci_lo, ci_hi, z, p, tau2, se_tau2, q, i2, h2
                let vals = [
                    r.beta,
                    r.se,
                    r.ci_lo,
                    r.ci_hi,
                    r.z,
                    r.p,
                    r.tau2,
                    r.se_tau2,
                    r.q,
                    r.i2,
                    r.h2,
                ];
                for (c, val) in vals.iter().enumerate() {
                    let vec = duckdb_data_chunk_get_vector(chunk.as_raw(), (4 + c) as u64);
                    let mut w = VectorWriter::new(vec);
                    if val.is_finite() {
                        w.write_f64(0, *val);
                    } else {
                        w.set_null(0);
                    }
                }
                duckdb_data_chunk_set_size(chunk.as_raw(), 1);
            }
            Ok(())
        })
        .build()
}

// ── meta_forest() ──────────────────────────────────────────────────

/// one row per study (missing dropped); tau2 from REML for random weights.
struct ForestRow {
    y: f64,
    v: f64,
    ci_fe_lo: f64,
    ci_fe_hi: f64,
    w_fe: f64,
    ci_re_lo: f64,
    ci_re_hi: f64,
    w_re: f64,
}

struct ForestState {
    rows: Vec<ForestRow>,
    offset: usize,
}

fn build_meta_forest() -> Result<TableFunctionBuilder, ExtensionError> {
    TableFunctionBuilder::new("meta_forest")
        .param_logical(LogicalType::list(TypeId::Double)) // y
        .param_logical(LogicalType::list(TypeId::Double)) // se
        .with_state::<ForestState, _>(|bind| {
            let y_all = read_list_param(bind, 0);
            let se_all = read_list_param(bind, 1);
            if y_all.len() != se_all.len() {
                return Err(ExtensionError::new(
                    "meta_forest: y and se must have the same length",
                ));
            }
            // meta_pool expects sampling variances v = se^2 (metafor's `vi`)
            let v_all: Vec<f64> = se_all.iter().map(|s| s * s).collect();
            let kept: Vec<(f64, f64)> = y_all
                .iter()
                .zip(&v_all)
                .filter(|(&yi, &vi)| yi.is_finite() && vi.is_finite() && vi > 0.0)
                .map(|(&yi, &vi)| (yi, vi))
                .collect();
            if kept.len() < 2 {
                return Err(ExtensionError::new(
                    "meta_forest: need at least 2 studies with finite y and positive se",
                ));
            }
            // REML tau2 for random-effects weights (JASP forest default)
            let ys: Vec<f64> = kept.iter().map(|t| t.0).collect();
            let vs: Vec<f64> = kept.iter().map(|t| t.1).collect();
            let pool = meta_pool(&ys, &vs, "REML", 0.95, 100, "z")
                .expect("forest: pool just validated");
            let tau2 = pool.tau2;
            let qz = qnorm(0.975);
            let sum_w_fe: f64 = kept.iter().map(|t| 1.0 / t.1).sum();
            let sum_w_re: f64 = kept.iter().map(|t| 1.0 / (t.1 + tau2)).sum();
            let rows: Vec<ForestRow> = kept
                .iter()
                .map(|&(yi, vi)| {
                    let wfe = 1.0 / vi;
                    let wre = 1.0 / (vi + tau2);
                    ForestRow {
                        y: yi,
                        v: vi,
                        ci_fe_lo: yi - qz * vi.sqrt(),
                        ci_fe_hi: yi + qz * vi.sqrt(),
                        w_fe: wfe / sum_w_fe,
                        ci_re_lo: yi - qz * (vi + tau2).sqrt(),
                        ci_re_hi: yi + qz * (vi + tau2).sqrt(),
                        w_re: wre / sum_w_re,
                    }
                })
                .collect();
            bind.add_result_column("study", TypeId::BigInt);
            bind.add_result_column("y", TypeId::Double);
            bind.add_result_column("v", TypeId::Double);
            bind.add_result_column("ci_fe_lo", TypeId::Double);
            bind.add_result_column("ci_fe_hi", TypeId::Double);
            bind.add_result_column("w_fe", TypeId::Double);
            bind.add_result_column("ci_re_lo", TypeId::Double);
            bind.add_result_column("ci_re_hi", TypeId::Double);
            bind.add_result_column("w_re", TypeId::Double);
            bind.set_cardinality(rows.len() as u64, true);
            Ok(ForestState { rows, offset: 0 })
        })
        .scan(|state, chunk| {
            if state.offset >= state.rows.len() {
                unsafe { duckdb_data_chunk_set_size(chunk.as_raw(), 0) };
                return Ok(());
            }
            let n = 2048usize.min(state.rows.len() - state.offset);
            unsafe {
                let raw = chunk.as_raw();
                let mut cols = [0u64; 9];
                for c in 0..9 {
                    cols[c] = c as u64;
                }
                for i in 0..n {
                    let r = &state.rows[state.offset + i];
                    let vals = [
                        (i as i64 + 1) as f64,
                        r.y,
                        r.v,
                        r.ci_fe_lo,
                        r.ci_fe_hi,
                        r.w_fe,
                        r.ci_re_lo,
                        r.ci_re_hi,
                        r.w_re,
                    ];
                    for c in 0..9 {
                        let vec = duckdb_data_chunk_get_vector(raw, cols[c]);
                        let mut w = VectorWriter::new(vec);
                        if c == 0 {
                            w.write_i64(i, i as i64 + 1);
                        } else {
                            w.write_f64(i, vals[c]);
                        }
                    }
                }
                duckdb_data_chunk_set_size(raw, n as idx_t);
            }
            state.offset += n;
            Ok(())
        })
        .build()
}

// ── registration ──────────────────────────────────────────────────

fn register_all(reg: &impl Registrar) -> Result<(), ExtensionError> {
    unsafe {
        reg.register_table(build_meta()?)?;
        reg.register_table(build_meta_forest()?)?;
    }
    Ok(())
}

// Entry point — DuckDB expects the symbol `{EXTENSION_NAME}_init_c_api`
// (EXTENSION_NAME = meta_ext per Makefile / Cargo [lib]).
#[no_mangle]
pub unsafe extern "C" fn meta_ext_init_c_api(
    info: duckdb_extension_info,
    access: *const duckdb_extension_access,
) -> bool {
    unsafe {
        init_extension_v2(info, access, quack_rs::DUCKDB_API_VERSION, |con| register_all(con))
    }
}

