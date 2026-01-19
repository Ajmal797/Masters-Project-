# app.py
# =============================================================================
# Flask wrapper for:
#   Part A: Random Forest diagnostics (scatter + time-agg line plots + species_r2.csv)
#   Part B: Global LSTM (load saved weights, make forecasts + metrics)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

# ---------- HARD DISABLE GPU & XLA BEFORE importing tensorflow ----------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = '--xla_gpu_cuda_data_dir=""'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from flask import Flask, render_template_string

# =============================================================================
# GLOBAL CONFIG (shared CSV)
# =============================================================================
CSV_PATH = r"C:\UMASS_D\SMAST_RA\flask\catch_fleet_added_depth.csv"

# =============================================================================
# PART A — RANDOM FOREST DIAGNOSTICS
# =============================================================================

RF_MODEL_PATH = r"C:\UMASS_D\SMAST_RA\flask\artifacts\rf_catch_model.pkl"
CATEGORICAL_COLS = ["species", "fleet", "polygon", "depth_category"]

RF_SCATTER_DIR = "species_plots"
RF_TIME_AGG_DIR = "species_plots_time_agg"
RF_R2_CSV = "species_r2.csv"

AGG = "sum"          # "sum" for totals; "mean" for average per year
MARKERSIZE = 4


def run_random_forest_diagnostics():
    """Run RF scatter per species + time-agg line plots + species_r2.csv."""
    # -------------------
    # 1) Load raw data + encoded copy
    # -------------------
    df_raw = pd.read_csv(CSV_PATH)   # keep strings for labels, plots

    # Make a copy for model input (numeric)
    df_enc = df_raw.copy()

    # Encode categoricals EXACTLY like training: category → codes
    for col in CATEGORICAL_COLS:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].astype("category")
            df_enc[col] = df_enc[col].cat.codes

    # -------------------
    # 2) Load saved RF model (weights)
    # -------------------
    rf_model = joblib.load(RF_MODEL_PATH)
    print("✅ Loaded Random Forest model from:", RF_MODEL_PATH)

    # ============================================================
    # A1) PER-SPECIES SCATTER PLOTS: Actual vs Predicted
    # ============================================================
    os.makedirs(RF_SCATTER_DIR, exist_ok=True)

    species_list = df_raw["species"].unique()

    for species_name in species_list:
        mask = df_raw["species"] == species_name

        # Encoded slice for prediction
        df_species_enc = df_enc.loc[mask].copy()
        y_true = df_species_enc["atoutput"].values
        X_species = df_species_enc.drop(columns=["atoutput"])

        y_pred = rf_model.predict(X_species)

        r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else np.nan

        # For plotting, attach predictions to RAW slice (for nice labels if needed)
        df_species_plot = df_raw.loc[mask].copy()
        df_species_plot["predicted_atoutput"] = y_pred

        # Scatter plot
        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, s=25)

        lo = np.min([y_true.min(), y_pred.min()])
        hi = np.max([y_true.max(), y_pred.max()])
        plt.plot([lo, hi], [lo, hi], "r--", lw=2)

        plt.title(f"Actual vs Predicted: {species_name}\nR² = {r2:.3f}")
        plt.xlabel("Actual atoutput")
        plt.ylabel("Predicted atoutput")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        safe_name = species_name.replace(" ", "_").replace("/", "_")
        plt.savefig(f"{RF_SCATTER_DIR}/{safe_name}.png", dpi=150)
        plt.close()

    print(f"✅ Saved {len(species_list)} scatter plots in '{RF_SCATTER_DIR}/' folder.")

    # ============================================================
    # A2) PER-SPECIES TIME-AGGREGATED LINE PLOTS + species_r2.csv
    # ============================================================
    os.makedirs(RF_TIME_AGG_DIR, exist_ok=True)

    data_raw = df_raw.copy()
    data_enc = df_enc.copy()

    # Make a safe year column from "time"
    try:
        data_raw["year"] = data_raw["time"].astype(int)
    except Exception:
        data_raw["time"] = pd.to_datetime(data_raw["time"])
        data_raw["year"] = data_raw["time"].dt.year.astype(int)

    species_list = data_raw["species"].unique()
    records = []

    for species_name in species_list:
        mask = data_raw["species"] == species_name

        sp_enc = data_enc.loc[mask].copy()
        y_true = sp_enc["atoutput"].values
        X_sp = sp_enc.drop(columns=["atoutput"])
        preds = rf_model.predict(X_sp).astype(float)

        sp_plot = data_raw.loc[mask].copy()
        sp_plot["pred"] = preds

        # Aggregate per year
        if AGG == "sum":
            agg = sp_plot.groupby("year", as_index=True).agg(
                actual=("atoutput", "sum"),
                predicted=("pred", "sum"),
            ).sort_index()
        else:
            agg = sp_plot.groupby("year", as_index=True).agg(
                actual=("atoutput", "mean"),
                predicted=("pred", "mean"),
            ).sort_index()

        if len(agg) == 0:
            continue

        full_years = pd.Index(
            range(int(agg.index.min()), int(agg.index.max()) + 1),
            name="year"
        )
        agg = agg.reindex(full_years)

        # Compute R² on non-missing years
        valid = agg.dropna()
        if len(valid) >= 2 and np.var(valid["actual"].values) > 0:
            r2 = r2_score(valid["actual"].values, valid["predicted"].values)
        else:
            r2 = np.nan

        records.append({
            "species": species_name,
            "n_years": len(valid),
            "r2": r2,
        })

        # Plot lines
        plt.figure(figsize=(14, 4.2))
        plt.plot(
            agg.index, agg["actual"],
            "-o", linewidth=2, markersize=MARKERSIZE, label="Actual"
        )
        plt.plot(
            agg.index, agg["predicted"],
            "-x", linewidth=2, markersize=MARKERSIZE, label="Predicted (RF)"
        )

        plt.title(f"{species_name} — Actual vs Predicted over Time\nR²={r2:.3f}")
        plt.xlabel("Year")
        plt.ylabel("atoutput")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        safe = species_name.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(RF_TIME_AGG_DIR, f"{safe}_actual_vs_pred_time.png"), dpi=150)
        plt.close()

    # Save RF R² per species
    r2_df = pd.DataFrame(records).sort_values("r2", ascending=False)
    r2_df.to_csv(RF_R2_CSV, index=False)

    print(f"✅ Saved time-agg plots to '{RF_TIME_AGG_DIR}/' and R² table to '{RF_R2_CSV}'.")
    print(r2_df.head(10))


# =============================================================================
# PART B — GLOBAL LSTM (LOAD WEIGHTS, METRICS + FORECASTS)
# =============================================================================

LSTM_SAVE_DIR   = Path(r"C:\UMASS_D\SMAST_RA\flask\global_kt_lstm_cpu_trail_2")
LSTM_PLOTS_DIR  = LSTM_SAVE_DIR / "new"
LSTM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LSTM_MODEL_WEIGHTS_PATH = LSTM_SAVE_DIR / "global_lstm_best.weights.h5"

SPECIES_COL = "species"
TIME_COL    = "time"
VALUE_COL   = "atoutput"

BASE_YEAR    = 1957.0
TRAIN_CUTOFF = 2013.0
FORECAST_END = 2035.0
REPORT_FROM  = 2014.0

LOOKBACK       = 24
MIN_SERIES_LEN = max(LOOKBACK + 10, 20)

BEST = {
    "emb_dim": 8,
    "units1": 128,
    "use_second": False,
    "units2": 96,
    "l2": 1.6308979287756594e-07,
    "dropout": 0.0,
    "lr": 0.0014538006339000467,
}

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True


def modal_step(times: np.ndarray) -> float:
    diffs = np.diff(np.unique(times))
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return 1.0
    rounded = np.round(diffs, 4)
    mc = Counter(rounded).most_common()
    maxc = mc[0][1]
    cands = sorted([v for v, c in mc if c == maxc])
    return float(cands[0])


def add_season_cols(df_step: pd.DataFrame, year_float_col: str = "year_float"):
    frac = np.modf(df_step[year_float_col].values)[0].astype(np.float32)
    df_step["sin_2pi"] = np.sin(2 * np.pi * frac).astype(np.float32)
    df_step["cos_2pi"] = np.cos(2 * np.pi * frac).astype(np.float32)
    return df_step


def build_windows(y_log: np.ndarray, sinv: np.ndarray, cosv: np.ndarray, L: int):
    Xs, ys = [], []
    for t in range(L, len(y_log)):
        Xs.append(np.stack([y_log[t-L:t], sinv[t-L:t], cosv[t-L:t]], axis=1))
        ys.append(y_log[t])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(y_true == 0.0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def plot_history_and_forecast(species_name: str,
                              year_hist: np.ndarray, y_hist: np.ndarray,
                              year_fcst: np.ndarray, y_fcst: np.ndarray,
                              outpath: Path):
    plt.figure()
    plt.plot(year_hist, y_hist, label="History (≤2013)", linewidth=2)
    if len(year_fcst) > 0:
        plt.plot(year_fcst, y_fcst, label="Forecast (≥2014→2035)", linewidth=2)
    plt.title(f"{species_name} — Global LSTM (from saved weights)")
    plt.xlabel("calendar year (fractional)")
    plt.ylabel("atoutput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def build_fixed_model(lookback: int, n_features: int, n_species: int) -> tf.keras.Model:
    seq_in = layers.Input(shape=(lookback, n_features), name="seq_in")
    sp_id  = layers.Input(shape=(1,), dtype="int32", name="species_id")

    emb_dim = int(BEST["emb_dim"])
    sp_emb  = layers.Embedding(
        input_dim=n_species,
        output_dim=emb_dim,
        name="sp_emb"
    )(sp_id)
    sp_emb  = layers.Reshape((1, emb_dim))(sp_emb)
    sp_emb  = layers.Lambda(lambda x: tf.repeat(x, repeats=lookback, axis=1))(sp_emb)

    x = layers.Concatenate(axis=-1)([seq_in, sp_emb])

    units1     = int(BEST["units1"])
    use_second = bool(BEST["use_second"])
    l2val      = float(BEST["l2"])
    dropout    = float(BEST["dropout"])
    reg = regularizers.l2(l2val)

    x = layers.LSTM(
        units1,
        return_sequences=use_second,
        kernel_regularizer=reg
    )(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    if use_second:
        units2 = int(BEST["units2"])
        x = layers.LSTM(
            units2,
            return_sequences=False,
            kernel_regularizer=reg
        )(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(1)(x)

    model = models.Model(inputs=[seq_in, sp_id], outputs=out)
    lr = float(BEST["lr"])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse")
    return model


def run_lstm_from_saved_weights():
    """Load LSTM weights, compute per-species metrics + forecasts + plots."""
    if not LSTM_MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Saved weights not found at {LSTM_MODEL_WEIGHTS_PATH}. "
            "Run the LSTM training script first."
        )

    df = pd.read_csv(CSV_PATH)
    need = {SPECIES_COL, TIME_COL, VALUE_COL}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain {need}")

    df["time_frac"]  = df[TIME_COL].astype(float)
    df["year_float"] = BASE_YEAR + df["time_frac"]
    agg = (
        df.groupby([SPECIES_COL, "time_frac"], as_index=False)[VALUE_COL]
          .sum()
          .rename(columns={VALUE_COL: "y"})
          .sort_values(["species", "time_frac"])
    )
    agg["year_float"] = BASE_YEAR + agg["time_frac"]

    species_list = agg[SPECIES_COL].unique().tolist()
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    n_species = len(species_list)

    per_sp = {}
    for sp in species_list:
        sdf = agg[agg[SPECIES_COL] == sp].copy().sort_values("year_float")
        per_sp[sp] = sdf

    # Safe clear_session that won't crash if backend internals change
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        print("Warning: could not clear Keras session:", e)

    model = build_fixed_model(LOOKBACK, n_features=3, n_species=n_species)
    print(f"🔹 Loading LSTM weights from {LSTM_MODEL_WEIGHTS_PATH}")
    model.load_weights(str(LSTM_MODEL_WEIGHTS_PATH))

    # ---- Per-species validation metrics ----
    per_metrics = []

    for sp in species_list:
        sdf = per_sp[sp]
        sdf = add_season_cols(sdf, "year_float")
        tr_mask = sdf["year_float"] <= TRAIN_CUTOFF
        sdf_tr = sdf[tr_mask].copy()
        if len(sdf_tr) < MIN_SERIES_LEN:
            continue

        y_hist   = sdf_tr["y"].astype(float).values
        yr_hist  = sdf_tr["year_float"].values
        sin_hist = sdf_tr["sin_2pi"].values
        cos_hist = sdf_tr["cos_2pi"].values
        y_log    = np.log1p(np.clip(y_hist, 0.0, None))

        X_sp, y_sp = build_windows(y_log, sin_hist, cos_hist, LOOKBACK)
        if len(X_sp) < 10:
            continue

        v = max(1, int(round(0.2 * len(X_sp))))
        Xva_sp, yva_sp = X_sp[-v:], y_sp[-v:]
        spid_sp = np.full((len(Xva_sp), 1), species_to_idx[sp], np.int32)

        if len(Xva_sp) == 0:
            continue

        yva_pred_log = model.predict([Xva_sp, spid_sp], verbose=0).ravel()
        yva_true = np.expm1(yva_sp)
        yva_pred = np.expm1(yva_pred_log)
        rmse = float(math.sqrt(mean_squared_error(yva_true, yva_pred)))
        mae  = float(mean_absolute_error(yva_true, yva_pred))
        mape = float(safe_mape(yva_true, yva_pred))
        r2   = float(r2_score(yva_true, yva_pred)) if len(np.unique(yva_true)) > 1 else float("nan")

        per_metrics.append({
            "species": sp,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_%": mape,
            "R2": r2,
            "val_points": int(len(yva_true))
        })
        print(f"✅ LSTM Val {sp}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # ---- Forecast ≥2014→2035 per species ----
    per_forecasts = []
    for sp in species_list:
        sdf = per_sp[sp].copy().sort_values("year_float")
        sdf = add_season_cols(sdf, "year_float")
        sdf_tr = sdf[sdf["year_float"] <= TRAIN_CUTOFF].copy()
        if len(sdf_tr) < MIN_SERIES_LEN:
            continue

        y_hist  = sdf_tr["y"].astype(float).values
        yrs_tr  = sdf_tr["year_float"].values
        y_log   = np.log1p(np.clip(y_hist, 0.0, None))
        frac_tr = np.modf(yrs_tr)[0].astype(np.float32)

        step_sp     = modal_step(sdf["time_frac"].values)
        step_global = modal_step(agg["time_frac"].values)
        step = step_sp if step_sp > 1e-9 else step_global

        last_tr = yrs_tr[-1]
        fut_years = last_tr + np.arange(1, int(np.ceil((FORECAST_END - last_tr)/step)) + 1) * step
        fut_years = fut_years[fut_years <= FORECAST_END + 1e-9]
        if len(fut_years) == 0:
            continue
        fut_fracs = np.modf(fut_years)[0].astype(np.float32)

        ywin = y_log[-LOOKBACK:].copy()
        fwin = frac_tr[-LOOKBACK:].copy()
        preds_log = []
        for f in fut_fracs:
            s_hist = np.sin(2*np.pi*fwin).astype(np.float32)
            c_hist = np.cos(2*np.pi*fwin).astype(np.float32)
            Xwin = np.stack([ywin, s_hist, c_hist], axis=1)[None, ...]
            sp_in = np.array([[species_to_idx[sp]]], dtype=np.int32)
            nxt = float(model.predict([Xwin, sp_in], verbose=0)[0, 0])
            preds_log.append(nxt)
            ywin = np.concatenate([ywin[1:], np.array([nxt], np.float32)])
            fwin = np.concatenate([fwin[1:], np.array([f],  np.float32)])
        preds = np.expm1(np.array(preds_log, np.float32))

        mask = fut_years >= REPORT_FROM
        for k, yf in enumerate(fut_years[mask]):
            per_forecasts.append({
                "species": sp,
                "year_float": float(yf),
                "forecast_atoutput": float(preds[mask][k])
            })

        plot_history_and_forecast(
            sp, yrs_tr, y_hist,
            fut_years[mask], preds[mask],
            LSTM_PLOTS_DIR / f"{sp.replace('/', '_')}.png"
        )

    # ---- Save CSVs & summary ----
    if per_metrics:
        pd.DataFrame(per_metrics).sort_values("RMSE").to_csv(
            LSTM_SAVE_DIR / "metrics_per_species.csv", index=False
        )
    if per_forecasts:
        pd.DataFrame(per_forecasts).to_csv(
            LSTM_SAVE_DIR / "forecast_per_species.csv", index=False
        )

    if per_metrics:
        weights = np.array([m["val_points"] for m in per_metrics], float)
        weights = np.where(weights <= 0, 1.0, weights)
        rmses = np.array([m["RMSE"] for m in per_metrics], float)
        maes  = np.array([m["MAE"]  for m in per_metrics], float)
        mapes = np.array([m["MAPE_%"] for m in per_metrics], float)
        r2s   = np.array([m["R2"] for m in per_metrics], float)
        w = weights / weights.sum()
        overall = (
            f"Species: {len(per_metrics)}\n"
            f"Weighted RMSE: {np.sum(w * rmses):.4f}\n"
            f"Weighted MAE : {np.sum(w * maes):.4f}\n"
            f"Weighted MAPE: {np.sum(w * mapes):.2f}%\n"
            f"Mean R2 (finite): {np.nanmean(r2s):.4f}\n"
        )
    else:
        overall = "No species produced validation metrics.\n"

    print("\n===== LSTM OVERALL SUMMARY =====\n" + overall)
    with open(LSTM_SAVE_DIR / "overall_summary.txt", "w") as f:
        f.write(overall)


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>Catch Fleet Emulator – RF + LSTM</title>
</head>
<body>
  <h1>Catch Fleet Emulator</h1>
  <p>This app runs two parts on your dataset:</p>
  <ul>
    <li><b>Part A</b>: Random Forest diagnostics (scatter plots, time-agg plots, species_r2.csv)</li>
    <li><b>Part B</b>: Global LSTM (load saved weights, per-species metrics, forecasts, plots)</li>
  </ul>
  <form method="post" action="/run">
    <button type="submit">Run RF + LSTM Pipeline</button>
  </form>
  <p>Outputs will be written to these folders/files (relative to app.py):</p>
  <ul>
    <li><code>species_plots/</code></li>
    <li><code>species_plots_time_agg/</code></li>
    <li><code>species_r2.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/new/</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/metrics_per_species.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/forecast_per_species.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/overall_summary.txt</code></li>
  </ul>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
<head>
  <title>Catch Fleet Emulator – Finished</title>
</head>
<body>
  <h1>Pipeline Finished</h1>
  <p>{{ msg }}</p>
  <p>Check these folders/files on disk:</p>
  <ul>
    <li><code>species_plots/</code></li>
    <li><code>species_plots_time_agg/</code></li>
    <li><code>species_r2.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/new/</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/metrics_per_species.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/forecast_per_species.csv</code></li>
    <li><code>global_kt_lstm_cpu_trail_2/overall_summary.txt</code></li>
  </ul>
  <p><a href="/">Back to Home</a></p>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/run", methods=["POST"])
def run_pipeline():
    try:
        print("========== PART A: Random Forest diagnostics ==========")
        run_random_forest_diagnostics()

        print("\n========== PART B: LSTM (from saved weights) ==========")
        run_lstm_from_saved_weights()

        msg = "RF diagnostics + LSTM forecasting completed successfully."
    except Exception as e:
        msg = f"Error during pipeline: {e}"
        print(msg)

    return render_template_string(RESULT_HTML, msg=msg)


if __name__ == "__main__":
    # Access in browser at http://127.0.0.1:5000
    app.run(debug=True)
