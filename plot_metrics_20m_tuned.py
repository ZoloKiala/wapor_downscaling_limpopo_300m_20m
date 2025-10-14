# -*- coding: utf-8 -*-
# plot_metrics_20m_tuned.py — Local Machine Edition
# ------------------------------------------------------------
# Plots time-series metrics and a global Pred vs Native scatter
# for RF downscaling outputs written by outputs_rf20_tuned/.
#
# Usage (defaults to tuned outputs):
#   python plot_metrics_20m_tuned.py
#
# Optional: override paths via env vars
#   DATA_DIR=wapor_20m_local OUT_DIR=wapor_20m_local/outputs_rf20_tuned python plot_metrics_20m_tuned.py
# ------------------------------------------------------------

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

# ---- CONFIG (tuned defaults) ----
DATA_DIR = os.getenv("DATA_DIR", "wapor_20m_local")
OUT_DIR  = os.getenv("OUT_DIR",  os.path.join(DATA_DIR, "outputs_rf20_tuned"))
CSV_PATH = os.getenv("CSV_PATH", os.path.join(OUT_DIR, "metrics_tuned.csv"))
PRED_DIR = os.getenv("PRED_DIR", os.path.join(OUT_DIR, "preds"))

def _parse_date_from_tag(tag: str):
    if not isinstance(tag, str):
        return None
    m = re.search(r'(\d{4}-\d{2}-\d{2})', tag)
    if not m:  # fallback for 20240101 style
        m = re.search(r'(\d{4})(\d{2})(\d{2})', tag)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return None
    return m.group(1)

def _ensure_date(df: pd.DataFrame):
    if "date" not in df.columns:
        df["date"] = df.get("tag", "").apply(_parse_date_from_tag)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    return df

def _ensure_numeric(df: pd.DataFrame, cols=("n","mae","rmse","bias","r","r2")):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "n" not in df.columns:
        df["n"] = np.nan
    return df

def plot_time_series(csv_path=CSV_PATH, out_png=None, title="(Tuned) Prediction Accuracy over Time"):
    df = pd.read_csv(csv_path)
    df = _ensure_date(df)
    df = _ensure_numeric(df)

    # weighted metrics by valid pixel count n
    w = df["n"].fillna(0).values
    def wavg(col):
        if col not in df.columns: return np.nan
        x = df[col].values
        m = np.isfinite(x) & np.isfinite(w)
        if not m.any() or w[m].sum() <= 0: return np.nan
        return np.average(x[m], weights=w[m])

    overall = {
        "R2_w":   wavg("r2"),
        "MAE_w":  wavg("mae"),
        "RMSE_w": wavg("rmse"),
        "Bias_w": wavg("bias"),
    }
    print("Weighted-overall metrics:", {k: (None if not np.isfinite(v) else round(v,3)) for k,v in overall.items()})

    # --- Plots (R2, MAE, RMSE) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if "r2" in df.columns:
        axes[0].plot(df["date"], df["r2"], marker="o", lw=1)
        axes[0].set_ylabel("R²")
        if np.isfinite(overall["R2_w"]):
            axes[0].axhline(overall["R2_w"], ls="--", lw=1, color="gray", label=f"weighted R²={overall['R2_w']:.2f}")
            axes[0].legend(loc="lower right")
        axes[0].grid(True, linestyle=":")

    if "mae" in df.columns:
        axes[1].plot(df["date"], df["mae"], marker="o", lw=1)
        axes[1].set_ylabel("MAE")
        axes[1].grid(True, linestyle=":")

    if "rmse" in df.columns:
        axes[2].plot(df["date"], df["rmse"], marker="o", lw=1)
        axes[2].set_ylabel("RMSE")
        axes[2].grid(True, linestyle=":")
        axes[2].set_xlabel("Dekad")

    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    if out_png is None:
        out_png = os.path.join(OUT_DIR, "accuracy_timeseries_tuned.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved:", out_png)
    return out_png, overall

def _guess_pair_paths(row):
    pth_p = row.get("pred_clip")
    pth_t = row.get("label_clip")
    if isinstance(pth_p, str) and isinstance(pth_t, str):
        return pth_p, pth_t
    tag = row.get("tag")
    if not isinstance(tag, str):
        return None, None
    pred = os.path.join(PRED_DIR, f"{tag}_pred_clip.tif")
    lab  = os.path.join(PRED_DIR, f"{tag}_label_clip.tif")
    if os.path.exists(pred) and os.path.exists(lab):
        return pred, lab
    return None, None

def scatter_pred_vs_true(csv_path=CSV_PATH, sample_per_file=3000, out_png=None,
                         title="(Tuned) Pred vs. Native (sampled)"):
    df = pd.read_csv(csv_path)
    xs, ys = [], []
    for _, r in df.iterrows():
        pth_p, pth_t = _guess_pair_paths(r)
        if not pth_p or not pth_t:
            continue
        if (not os.path.exists(pth_p)) or (not os.path.exists(pth_t)):
            continue
        try:
            with rasterio.open(pth_p) as dp, rasterio.open(pth_t) as dt:
                p = dp.read(1).astype(np.float32)
                t = dt.read(1).astype(np.float32)
                m = np.isfinite(p) & np.isfinite(t)
                ndp, ndt = dp.nodata, dt.nodata
                if ndp is not None: m &= (p != ndp)
                if ndt is not None: m &= (t != ndt)
                if m.sum() == 0: continue
                yy, xx = np.where(m)
                k = min(sample_per_file, len(yy))
                if k <= 0: continue
                sel = np.random.choice(len(yy), size=k, replace=False)
                xs.append(t[yy[sel], xx[sel]])
                ys.append(p[yy[sel], xx[sel]])
        except Exception as e:
            print("skip scatter on", pth_p, "->", e)

    if not xs:
        print("No valid raster pairs found for scatter.")
        return None

    X = np.concatenate(xs)
    Y = np.concatenate(ys)

    qstack = np.concatenate([X, Y])
    vmin = float(np.percentile(qstack, 2))
    vmax = float(np.percentile(qstack, 98))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin >= vmax:
        vmin, vmax = 0.0, 8.0

    r = float(np.corrcoef(X, Y)[0,1])
    r2 = r*r
    mae = float(np.mean(np.abs(Y - X)))

    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, s=2, alpha=0.25)
    plt.plot([vmin, vmax],[vmin, vmax], 'k--', lw=1)
    plt.xlim(vmin, vmax); plt.ylim(vmin, vmax)
    plt.xlabel("Native (label)")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nR²={r2:.2f}  MAE={mae:.2f}")
    plt.grid(True, linestyle=":", alpha=0.5)
    if out_png is None:
        out_png = os.path.join(OUT_DIR, "scatter_pred_vs_native_tuned.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved:", out_png)
    return out_png

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    ts_png, overall = plot_time_series(CSV_PATH)
    scatter_png = scatter_pred_vs_true(CSV_PATH, sample_per_file=3000)
    try:
        # Best effort show results if running in a notebook/IDE that supports it
        from IPython.display import Image, display
        display(Image(filename=ts_png))
        if scatter_png: display(Image(filename=scatter_png))
    except Exception:
        pass
