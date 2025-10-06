# -*- coding: utf-8 -*-
"""
WaPOR 20 m Downscaling — Local Machine Edition
- Exports dekadal 20 m stacks with geedim (Baixo / Lamego by default)
- Trains a RandomForest on training stacks and predicts on test stacks
- Saves metrics and visualisations

Notes:
- Requires a Google Earth Engine service account key JSON locally accessible.
  You can set the env var EE_SERVICE_ACCOUNT_FILE to point to it, or keep the default filename in this folder.
"""

import os
import re
import glob
import json
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import ee
from google.oauth2 import service_account
import geedim  # registers .gd on ee.Image / ee.ImageCollection

# ---------- AUTH (LOCAL) ----------
SERVICE_ACCOUNT_FILE = os.getenv("EE_SERVICE_ACCOUNT_FILE", "tethys-app-1-acc3960d3dd6.json")
SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive",
]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
ee.Initialize(credentials)

# ---------- CONFIG ----------
# AOIs (Earth Engine FeatureCollections)
BAIXO_AOI  = "projects/tethys-app-1/assets/baixo"
LAMEGO_AOI = "projects/tethys-app-1/assets/lamego"

# Label image collections (dekadal WaPOR L3 at 20 m, per-region)
L3_BAIXO_D_COLL  = "projects/tethys-app-1/assets/WaPOR_L3_20m_D_BAIXO"
L3_LAMEGO_D_COLL = "projects/tethys-app-1/assets/WaPOR_L3_20m_D_LAMEGO"
LABEL_BAND       = "b1"   # change if your label band name differs

# Windows
TRAIN_START = "2018-01-01"; TRAIN_END = "2020-12-01"
TEST_START  = "2019-01-01"; TEST_END  = "2021-10-01"

# Local output root
OUTPUT_ROOT = Path("wapor_20m_local")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Export / raster settings
EXPORT_SCALE       = 20
EXPORT_DTYPE       = "float32"
EXPORT_NODATA      = -9999
GEEDIM_TILE_MB     = 16   # must be < 32
GEEDIM_MAX_REQUEST = 16

# Predictors (EE sources)
S2_SR        = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
L1_AETI_D    = ee.ImageCollection("FAO/WAPOR/3/L1_AETI_D")
DEM          = ee.Image("USGS/SRTMGL1_003").rename("DEM")
SLOPE        = ee.Terrain.slope(DEM).rename("Slope")
LANDCOVER    = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("LC")

# Sentinel-1 & CHIRPS
S1_GRD       = (ee.ImageCollection("COPERNICUS/S1_GRD")
                 .filter(ee.Filter.eq("instrumentMode", "IW"))
                 .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
                 .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")))
CHIRPS_DAILY = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")  # mm/day

# --- Sentinel-1 / Texture toggles ---
S1_VALUES_ARE_DB = True   # COPERNICUS/S1_GRD is usually in dB; if not, set to False to convert
S1_SMOOTH_RADIUS = 1      # pixels; light speckle smoothing on dB
TEXTURE_SIZE     = 3      # window (pixels) for GLCM textures
DB_MIN, DB_MAX   = -25.0, 5.0  # range used to quantize to 8-bit for GLCM

# Map human-friendly names → EE glcmTexture suffixes
GLCM_MAP = {
    "contrast":      "contrast",
    "entropy":       "ent",
    "homogeneity":   "idm",   # inverse difference moment
    "dissimilarity": "diss",
}

# CHIRPS lag settings
CHIRPS_LAG_DAYS = 10

# AOIs
BAIXO  = ee.FeatureCollection(BAIXO_AOI)
LAMEGO = ee.FeatureCollection(LAMEGO_AOI)

# Label ICs
L3_BAIXO_D  = (ee.ImageCollection(L3_BAIXO_D_COLL)
               .filterBounds(BAIXO)
               .filterDate(TRAIN_START, TRAIN_END))
L3_LAMEGO_D = (ee.ImageCollection(L3_LAMEGO_D_COLL)
               .filterBounds(LAMEGO)
               .filterDate(TEST_START, TEST_END))

# ---------- HELPERS ----------
def end_of_dekad_from_label_start(date):
    """Exclusive end date for dekad starting at date (1st→11th→21st→next month)."""
    d = ee.Number(ee.Date(date).get('day'))
    is_third = d.gte(21)
    return ee.Date(ee.Algorithms.If(is_third, ee.Date(date).advance(1,'month'),
                                    ee.Date(date).advance(10,'day')))

def s2_dekad_safe(start, end, region):
    bnames = ["B2","B3","B4","B8","B11","B12"]
    win   = S2_SR.filterBounds(region).filterDate(start, end)
    win70 = win.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
    img70 = win70.median().select(bnames)
    imgAny= win.median().select(bnames)
    masked= ee.Image.constant([0]*len(bnames)).rename(bnames).updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(win70.size().gt(0), img70,
           ee.Algorithms.If(win.size().gt(0), imgAny, masked)))

def l1_dekad_safe(start, end, region):
    coll = L1_AETI_D.filterBounds(region).filterDate(start, end)
    img  = coll.mean().select([0]).rename("ETa300m")
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), img,
           ee.Image(0).updateMask(ee.Image(0)).rename("ETa300m")))

def to_db(img_lin):
    return img_lin.log10().multiply(10.0)

def s1_textures_from_db(vvdb, vhdb, size=TEXTURE_SIZE, db_min=DB_MIN, db_max=DB_MAX):
    """Compute GLCM textures on 8-bit quantized dB images."""
    def q8(img, name_u8):
        return (img.unitScale(db_min, db_max)
                    .multiply(255).clamp(0, 255).toUint8().rename(name_u8))
    vv_u8 = q8(vvdb, "S1_VV_u8")
    vh_u8 = q8(vhdb, "S1_VH_u8")

    vv_all = vv_u8.glcmTexture(size)
    vh_all = vh_u8.glcmTexture(size)

    vv_list = [vv_all.select(f"S1_VV_u8_{suf}").rename(f"S1_VV_{human}")
               for human, suf in GLCM_MAP.items()]
    vh_list = [vh_all.select(f"S1_VH_u8_{suf}").rename(f"S1_VH_{human}")
               for human, suf in GLCM_MAP.items()]
    return ee.Image.cat(vv_list + vh_list)

def s1_dekad_safe(start, end, region):
    """
    Median VV/VH over dekad (dB), light smoothing, VV−VH, and GLCM textures.
    Returns all S1 bands (including textures). Empty dekad → fully masked zeros.
    """
    coll = S1_GRD.filterBounds(region).filterDate(start, end)

    def _make():
        vv = coll.select("VV").median()
        vh = coll.select("VH").median()
        if S1_VALUES_ARE_DB:
            vvdb = vv
            vhdb = vh
        else:
            vvdb = to_db(vv)
            vhdb = to_db(vh)
        vvdb_s = vvdb.focal_mean(radius=S1_SMOOTH_RADIUS, units="pixels").rename("S1_VV_dB")
        vhdb_s = vhdb.focal_mean(radius=S1_SMOOTH_RADIUS, units="pixels").rename("S1_VH_dB")
        diff   = vvdb.subtract(vhdb).focal_mean(radius=S1_SMOOTH_RADIUS, units="pixels").rename("S1_VVminusVH_dB")
        tex    = s1_textures_from_db(vvdb, vhdb)
        return ee.Image.cat([vvdb_s, vhdb_s, diff, tex])

    placeholders = [
        "S1_VV_dB","S1_VH_dB","S1_VVminusVH_dB",
        "S1_VV_contrast","S1_VV_entropy","S1_VV_homogeneity","S1_VV_dissimilarity",
        "S1_VH_contrast","S1_VH_entropy","S1_VH_homogeneity","S1_VH_dissimilarity",
    ]
    masked = ee.Image.constant([0]*len(placeholders)).rename(placeholders).updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), _make(), masked))

def chirps_sum(start, end, band_name):
    coll = CHIRPS_DAILY.filterDate(start, end).select("precipitation")  # mm/day
    img  = coll.sum().rename(band_name)
    return ee.Image(ee.Algorithms.If(coll.size().gt(0),
                                     img, ee.Image(0).updateMask(ee.Image(0)).rename(band_name)))

def chirps_dekad_pair(start, end):
    cur  = chirps_sum(start, end, "CHIRPS10d")
    prev_start = ee.Date(start).advance(-CHIRPS_LAG_DAYS, "day")
    prev_end   = ee.Date(start)
    lag  = chirps_sum(prev_start, prev_end, "CHIRPS10d_lag")
    return cur.addBands(lag)

def build_stack_for_label(label_img, region_fc, include_label=True):
    """Return (stack, startDate) with fixed band order expected by local pipeline."""
    lbl   = ee.Image(label_img).select([0]).rename([LABEL_BAND])
    start = ee.Date(lbl.get("system:time_start"))
    end   = end_of_dekad_from_label_start(start)

    s2    = s2_dekad_safe(start, end, region_fc)
    ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
    l1    = l1_dekad_safe(start, end, region_fc)
    s1    = s1_dekad_safe(start, end, region_fc)
    rain2 = chirps_dekad_pair(start, end)

    stack = (s2.addBands([ndvi, l1, DEM, SLOPE, LANDCOVER, s1, rain2])
               .select([
                   "B4","B8","B11","NDVI","ETa300m",
                   "DEM","Slope","LC",
                   "S1_VV_dB","S1_VH_dB","S1_VVminusVH_dB",
                   "S1_VV_contrast","S1_VV_entropy","S1_VV_homogeneity","S1_VV_dissimilarity",
                   "S1_VH_contrast","S1_VH_entropy","S1_VH_homogeneity","S1_VH_dissimilarity",
                   "CHIRPS10d","CHIRPS10d_lag"
               ]))
    if include_label:
        stack = stack.addBands(lbl)
    return stack, start

# ---------- LOCAL EXPORT WITH GEEDIM ----------
def export_ic_local_geedim(label_ic, region_fc, region_name, subfolder,
                           include_label=True, scale=EXPORT_SCALE,
                           dtype=EXPORT_DTYPE, nodata_val=EXPORT_NODATA,
                           max_tile_size_mb=GEEDIM_TILE_MB, max_requests=GEEDIM_MAX_REQUEST):
    """
    Download each dekad stack as a single GeoTIFF to OUTPUT_ROOT/subfolder using geedim.
    """
    ic   = label_ic.sort("system:time_start")
    n    = ic.size().getInfo()
    lst  = ic.toList(n)
    geom = region_fc.geometry()

    out_dir = OUTPUT_ROOT / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img   = ee.Image(lst.get(i))
        stack, start = build_stack_for_label(img, region_fc, include_label=include_label)
        date_str = start.format("YYYY-MM-dd").getInfo()
        fname    = f"{region_name}_{date_str}_stack.tif"
        fpath    = str(out_dir / fname)

        prepared = (stack.clip(geom)
                         .gd.prepareForExport(region=geom, scale=scale, dtype=dtype))
        try:
            prepared.gd.toGeoTIFF(
                fpath,
                overwrite=True,
                nodata=nodata_val,
                max_tile_size=max_tile_size_mb,  # MB, must be < 32
                max_requests=max_requests
            )
            print("Saved:", fpath)
        except Exception as e:
            print(f"[WARN] Failed {fname}: {e}")

# ============================================================
# WaPOR 20 m Downscaling — RandomForest (auto features)
# ============================================================

# ------------------ PATHS ------------------
# Option A: one folder containing both BAIXO and LAMEGO stacks (default from exporter)
DATA_DIR   = "wapor_20m_local"  # contains subfolders BAIXO/ and LAMEGO/

# Option B: override explicit train/test folders (each with *stack*.tif files)
TRAIN_DIR  = None               # e.g. "wapor_20m_local/BAIXO"
TEST_DIR   = None               # e.g. "wapor_20m_local/LAMEGO"

OUT_DIR    = os.path.join(DATA_DIR if TRAIN_DIR is None else TRAIN_DIR, "outputs_rf20_auto")
PRED_DIR   = os.path.join(OUT_DIR, "preds")
VIS_DIR    = os.path.join(OUT_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(PRED_DIR, exist_ok=True); os.makedirs(VIS_DIR, exist_ok=True)

# ------------------ DATES (used when TRAIN_DIR/TEST_DIR are None) ------------------
# For your exporter, filenames look like: Region_YYYY-MM-DD_stack.tif
# Set windows for BAIXO (train) and LAMEGO (test), or let fallback 70/30 split handle it.
TRAIN_START = "2018-01-01"; TRAIN_END = "2018-01-21"   # [start, end)
TEST_START  = "2019-01-01"; TEST_END  = "2019-0-21"   # inclusive

# ------------------ SAMPLING / RF CONFIG ------------------
PER_FILE_SAMPLES  = 300
CHUNK_PRED        = 1024
RANDOM_STATE      = 7
NODATA_OUT        = -9999.0

# Feature options
DO_ONE_HOT_LC     = True
USE_DAY_OF_YEAR   = True
MAX_LC_UNIQUE     = 20

# Auto-feature discovery controls
MAX_SCAN_FILES    = 200      # scan up to N training files for band presence
PCT_REQUIRED      = 0.90     # keep predictors appearing in ≥90% of scanned training files

# Label band candidates (your exporter ends with label 'b1')
LABEL_CANDIDATES  = ("b1","ETa20","ETa20m","AETI20","L3","LABEL","label")

# Random Forest (strong but not overfit)
RF_PARAMS = dict(
    n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    bootstrap=True, max_samples=0.8, oob_score=True,
    warm_start=False
)

np.random.seed(RANDOM_STATE)

# ------------------ Helper utilities for RF ------------------
def parse_any_date(text):
    """
    Return (YYYY-MM-DD, DOY) parsed from filename.
    Supports YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD.
    """
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', text)
    if not m:
        m = re.search(r'(\d{4})(\d{2})(\d{2})', text)
    if not m:
        return None, None
    y, mo, d = map(int, m.groups())
    try:
        dte = dt.date(y, mo, d)
    except ValueError:
        return None, None
    doy = (dte - dt.date(y, 1, 1)).days + 1
    return dte.isoformat(), doy

def in_range(date_str, start, end, inclusive_end=False):
    if date_str is None: return False
    d = dt.date.fromisoformat(date_str)
    ds = dt.date.fromisoformat(start)
    de = dt.date.fromisoformat(end)
    return (ds <= d <= de) if inclusive_end else (ds <= d < de)

def discover_files():
    """
    Return (train_files, test_files).
    If TRAIN_DIR/TEST_DIR given → use them.
    Else: search in DATA_DIR subfolders BAIXO/, LAMEGO/; if missing, search DATA_DIR directly and split by date.
    """
    if TRAIN_DIR and TEST_DIR:
        train = sorted(glob.glob(os.path.join(TRAIN_DIR, "*stack*.tif")))
        test  = sorted(glob.glob(os.path.join(TEST_DIR , "*stack*.tif")))
        print(f"[DISCOVERY] TRAIN_DIR={TRAIN_DIR} -> {len(train)} files")
        print(f"[DISCOVERY] TEST_DIR ={TEST_DIR } -> {len(test)} files")
        return train, test

    # prefer BAIXO for train and LAMEGO for test if present
    baixo_dir  = os.path.join(DATA_DIR, "BAIXO")
    lamego_dir = os.path.join(DATA_DIR, "LAMEGO")
    if os.path.isdir(baixo_dir) and os.path.isdir(lamego_dir):
        train = sorted(glob.glob(os.path.join(baixo_dir, "*stack*.tif")))
        test  = sorted(glob.glob(os.path.join(lamego_dir, "*stack*.tif")))
        print(f"[DISCOVERY] BAIXO={len(train)} | LAMEGO={len(test)}")
        return train, test

    # otherwise, all in one folder → split by dates
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*stack*.tif")))
    print(f"[DISCOVERY] DATA_DIR={DATA_DIR}, found {len(files)} stack files")
    if not files:
        return [], []
    rows = []
    for fp in files:
        dstr, doy = parse_any_date(os.path.basename(fp))
        rows.append((fp, dstr, doy))
    df = pd.DataFrame(rows, columns=["fp","date","doy"])
    train = [r.fp for r in df.itertuples(index=False) if in_range(r.date, TRAIN_START, TRAIN_END, inclusive_end=False)]
    test  = [r.fp for r in df.itertuples(index=False) if in_range(r.date, TEST_START,  TEST_END,  inclusive_end=True)]
    if len(train) == 0:
        print("[DISCOVERY] No files in TRAIN window → using 70/30 split (by date).")
        dfx = df.dropna(subset=["date"]).copy()
        if dfx.empty:
            split = max(1, int(0.7 * len(files)))
            train, test = files[:split], files[split:]
        else:
            dfx["date_dt"] = pd.to_datetime(dfx["date"])
            dfx = dfx.sort_values("date_dt")
            split = max(1, int(0.7 * len(dfx)))
            train = dfx["fp"].iloc[:split].tolist()
            test  = dfx["fp"].iloc[split:].tolist()
    if len(test) == 0:
        rest = [f for f in files if f not in set(train)]
        test = rest
    print(f"[DISCOVERY] train: {len(train)} | test: {len(test)}")
    print(" sample train:", [os.path.basename(x) for x in train[:3]])
    print(" sample test :", [os.path.basename(x) for x in test[:3]])
    return train, test

def band_map(ds):
    """Map band name -> zero-based index using Rasterio band descriptions."""
    desc = list(ds.descriptions or [])
    return {desc[i]: i for i in range(ds.count) if i < len(desc) and desc[i]}

def guess_label_name(bm):
    for nm in LABEL_CANDIDATES:
        if nm in bm: return nm
    return None

def read_label_band(ds, bm=None):
    bm = bm or band_map(ds)
    nm = guess_label_name(bm)
    if nm is None:
        return ds.read(ds.count).astype(np.float32)  # fallback: last band
    return ds.read(bm[nm]+1).astype(np.float32)

# ------------------ Feature discovery ------------------
from collections import Counter

def discover_lc_categories(train_files, max_files=10, cap_unique=MAX_LC_UNIQUE):
    cats = set()
    for fp in train_files[:max_files]:
        with rasterio.open(fp) as ds:
            bm = band_map(ds)
            if "LC" not in bm: continue
            a = ds.read(bm["LC"]+1)
        vals = np.unique(a[np.isfinite(a)])
        vals = vals[(vals >= 0)]
        if vals.size:
            cats.update(np.unique(vals.astype(np.int32)).tolist())
    return sorted(list(cats))[:cap_unique]

def discover_predictor_names(train_files,
                             do_one_hot=DO_ONE_HOT_LC,
                             use_day=USE_DAY_OF_YEAR,
                             max_scan=MAX_SCAN_FILES,
                             pct_required=PCT_REQUIRED):
    """
    Scan training files: collect band names excluding label, keep those present in ≥ pct_required
    of scanned files. Add DOY and LC one-hot (expanded later) if requested.
    """
    counts = Counter()
    total  = 0
    for fp in train_files[:max_scan]:
        with rasterio.open(fp) as ds:
            bm = band_map(ds)
            if not bm: continue
            total += 1
            lbl = guess_label_name(bm)
            for nm in bm.keys():
                if nm != lbl:
                    counts[nm] += 1
    if total == 0:
        raise RuntimeError("No readable training files for feature discovery.")

    keep_raw = sorted([nm for nm, c in counts.items() if c / total >= pct_required])

    # Land cover OHE list (will expand names later)
    lc_values = None
    keep = keep_raw[:]
    if do_one_hot and "LC" in keep_raw:
        lc_values = discover_lc_categories(train_files, max_files=20, cap_unique=MAX_LC_UNIQUE)

    if use_day:
        keep += ["DAY_SIN","DAY_COS"]
    return keep, lc_values

# ------------------ Sampling / Matrix ------------------
def sample_from_stack(path, feature_names, n=2000, lc_values=None):
    with rasterio.open(path) as ds:
        bm = band_map(ds)
        label = read_label_band(ds, bm)
        nodata = ds.nodata
        mask = np.isfinite(label)
        if nodata is not None and np.isfinite(nodata):
            mask &= (label != nodata)

        needed_raw = set()
        for f in feature_names:
            if f.startswith("LC_"):
                needed_raw.add("LC"); continue
            if f in ("DAY_SIN","DAY_COS"):
                continue
            needed_raw.add(f)

        # load needed arrays; bail if a required band is missing
        arrs = {}
        for name in needed_raw:
            if name not in bm:
                # band not present -> skip this file
                return None, None
            arrs[name] = ds.read(bm[name]+1).astype(np.float32)

    # day-of-year
    base = os.path.basename(path)
    date_str, doy = parse_any_date(base)
    angle   = 2.0 * np.pi * ((doy or 1) / 365.0)
    day_sin = np.float32(np.sin(angle))
    day_cos = np.float32(np.cos(angle))

    # build validity mask
    for f in feature_names:
        if f.startswith("LC_"):
            if "LC" not in arrs: return None, None
            mask &= np.isfinite(arrs["LC"])
        elif f in ("DAY_SIN","DAY_COS"):
            pass
        else:
            a = arrs.get(f)
            if a is None: return None, None
            mask &= np.isfinite(a)

    ys, xs = np.where(mask)
    if ys.size == 0: return None, None
    take = min(n, ys.size)
    sel = np.random.choice(ys.size, size=take, replace=False)
    ys, xs = ys[sel], xs[sel]

    cols = []
    for f in feature_names:
        if f.startswith("LC_"):
            code = int(f.split("_",1)[1])
            lc = arrs["LC"][ys, xs].astype(np.int32)
            cols.append((lc == code).astype(np.float32))
        elif f == "DAY_SIN":
            cols.append(np.full_like(ys, day_sin, dtype=np.float32))
        elif f == "DAY_COS":
            cols.append(np.full_like(ys, day_cos, dtype=np.float32))
        else:
            cols.append(arrs[f][ys, xs])

    X = np.stack(cols, axis=1).astype(np.float32)
    y = label[ys, xs].astype(np.float32)
    return X, y

def build_training_matrix(train_files, feature_names, per_file_samples=2000, lc_values=None):
    Xs, ys = [], []
    used = 0
    for fp in train_files:
        Xi, yi = sample_from_stack(fp, feature_names, n=per_file_samples, lc_values=lc_values)
        if Xi is None:
            continue
        Xs.append(Xi); ys.append(yi); used += 1
    if not Xs:
        raise RuntimeError(f"No samples assembled for features={feature_names}. "
                           "Check band names / presence.")
    X = np.vstack(Xs); y = np.concatenate(ys)
    print(f"[TRAIN] Matrix: X={X.shape} from files={used}/{len(train_files)}")
    return X, y

# ------------------ Prediction / Post-process ------------------
def predict_stack_blockwise(fp_in, rf, feature_names, fp_out, lc_values=None,
                            chunk=CHUNK_PRED, nodata=NODATA_OUT):
    with rasterio.open(fp_in) as src:
        bm = band_map(src)
        H, W = src.height, src.width
        prof = src.profile.copy()
        out_prof = prof.copy()
        out_prof.update(count=1, dtype="float32", nodata=nodata,
                        compress="deflate", predictor=3, tiled=True)

        base = os.path.basename(fp_in)
        date_str, doy = parse_any_date(base)
        angle   = 2.0 * np.pi * ((doy or 1) / 365.0)
        day_sin = np.float32(np.sin(angle))
        day_cos = np.float32(np.cos(angle))

        with rasterio.open(fp_out, "w", **out_prof) as dst:
            for y0 in range(0, H, chunk):
                for x0 in range(0, W, chunk):
                    h0 = min(chunk, H - y0); w0 = min(chunk, W - x0)
                    needed = set()
                    for f in feature_names:
                        if f.startswith("LC_"): needed.add("LC")
                        elif f not in ("DAY_SIN","DAY_COS"): needed.add(f)

                    raw = {}
                    missing = False
                    for name in needed:
                        if name not in bm:
                            missing = True; break
                        raw[name] = src.read(bm[name]+1, window=Window(x0, y0, w0, h0)).astype(np.float32)
                    if missing:
                        dst.write(np.full((h0,w0), nodata, np.float32), 1, window=Window(x0, y0, w0, h0))
                        continue

                    feats = []
                    for f in feature_names:
                        if f.startswith("LC_"):
                            code = int(f.split("_",1)[1])
                            a = raw["LC"].astype(np.int32)
                            feats.append((a == code).astype(np.float32))
                        elif f == "DAY_SIN":
                            feats.append(np.full((h0,w0), day_sin, dtype=np.float32))
                        elif f == "DAY_COS":
                            feats.append(np.full((h0,w0), day_cos, dtype=np.float32))
                        else:
                            feats.append(raw[f])

                    block = np.stack(feats, axis=0)               # (F, h0, w0)
                    X = np.moveaxis(block, 0, -1).reshape(-1, len(feature_names))
                    valid = np.isfinite(X).all(axis=1)
                    pred = np.full((X.shape[0],), np.nan, np.float32)
                    if valid.any():
                        pred[valid] = rf.predict(X[valid])
                    pred = pred.reshape(h0, w0)
                    dst.write(pred, 1, window=Window(x0, y0, w0, h0))
    return fp_out

def clip_to_label_mask_pair(pred_tif, stack_tif, out_pred_tif, out_label_tif, nodata=NODATA_OUT):
    with rasterio.open(stack_tif) as ds:
        bm = band_map(ds)
        label = read_label_band(ds, bm)
        nod = ds.nodata
        mask = np.isfinite(label)
        if nod is not None and np.isfinite(nod): mask &= (label != nod)

    with rasterio.open(pred_tif) as dp:
        pred = dp.read(1).astype(np.float32)
        prof = dp.profile.copy()

    pred[~mask] = nodata
    ys, xs = np.where(mask)
    if ys.size == 0: raise RuntimeError("Label mask is empty.")

    ymin, ymax = ys.min(), ys.max()+1; xmin, xmax = xs.min(), xs.max()+1
    pred_c = pred[ymin:ymax, xmin:xmax]; lab_c = label[ymin:ymax, xmin:xmax]
    win = Window(xmin, ymin, xmax-xmin, ymax-ymin)
    new_transform = rasterio.windows.transform(win, dp.transform)
    new_prof = prof.copy()
    new_prof.update(height=pred_c.shape[0], width=pred_c.shape[1],
                    transform=new_transform, count=1, dtype="float32",
                    nodata=nodata, compress="deflate", predictor=3, tiled=True)
    with rasterio.open(out_pred_tif, "w", **new_prof) as d1: d1.write(pred_c, 1)
    with rasterio.open(out_label_tif, "w", **new_prof) as d2: d2.write(lab_c, 1)

def visualise_minmax_robust(pred_clip, label_clip, out_dir, tag, title_prefix="Region", make_diff=True):
    with rasterio.open(pred_clip) as dp, rasterio.open(label_clip) as dl:
        p = dp.read(1).astype(np.float32); t = dl.read(1).astype(np.float32)
        ndp, ndl = dp.nodata, dl.nodata
    valid = np.isfinite(p) & np.isfinite(t)
    if ndp is not None: valid &= (p != ndp)
    if ndl is not None: valid &= (t != ndl)
    pm = np.ma.array(p, mask=~valid); tm = np.ma.array(t, mask=~valid)

    # robust shared min/max
    vmin = float(np.min([pm.min(), tm.min()])) if pm.count() and tm.count() else 0.0
    vmax = float(np.max([pm.max(), tm.max()])) if pm.count() and tm.count() else 8.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax: vmin, vmax = 0.0, 8.0

    cmap = plt.cm.viridis.copy(); cmap.set_bad("white", 0.0)
    fig = plt.figure(figsize=(12,5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 0.06])
    axL = fig.add_subplot(gs[0,0]); axR = fig.add_subplot(gs[0,1]); cax = fig.add_subplot(gs[1,:])
    imL = axL.imshow(tm, cmap=cmap, vmin=vmin, vmax=vmax); axL.set_title(f"{title_prefix} — Native (label)"); axL.axis("off")
    imR = axR.imshow(pm, cmap=cmap, vmin=vmin, vmax=vmax); axR.set_title(f"{title_prefix} — Predicted"); axR.axis("off")
    cb = fig.colorbar(imR, cax=cax, orientation="horizontal"); cb.set_label("Target (e.g., AETI 20 m)")
    pair_png = os.path.join(out_dir, f"{title_prefix}_native_vs_pred_{tag}_minmax.png")
    fig.savefig(pair_png, dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    diff_png = None
    if pm.count()>0 and tm.count()>0 and make_diff:
        diff = np.ma.array(p - t, mask=~valid)
        if diff.count()>0:
            d98 = float(np.percentile(np.abs(diff.compressed()), 98))
            if np.isfinite(d98) and d98>0:
                fig = plt.figure(figsize=(6,5), constrained_layout=True)
                gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 0.08])
                ax = fig.add_subplot(gs[0,0]); cax = fig.add_subplot(gs[1,0])
                im = ax.imshow(diff, cmap="coolwarm", vmin=-d98, vmax=+d98); ax.set_title(f"{title_prefix} — Pred − Native"); ax.axis("off")
                cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Difference")
                diff_png = os.path.join(out_dir, f"{title_prefix}_pred_minus_native_{tag}_robust.png")
                fig.savefig(diff_png, dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    return {"pair_png": pair_png, "diff_png": diff_png}

def metrics_from_pair(pred_clip, label_clip):
    with rasterio.open(pred_clip) as dp, rasterio.open(label_clip) as dl:
        p = dp.read(1).astype(np.float32); t = dl.read(1).astype(np.float32)
        ndp, ndl = dp.nodata, dl.nodata
    m = np.isfinite(p) & np.isfinite(t)
    if ndp is not None: m &= (p != ndp)
    if ndl is not None: m &= (t != ndl)
    if m.sum() == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, r=np.nan, r2=np.nan)
    e = p[m] - t[m]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e*e)))
    bias = float(np.mean(e))
    r = float(np.corrcoef(p[m], t[m])[0,1])
    return dict(n=int(m.sum()), mae=mae, rmse=rmse, bias=bias, r=r, r2=r*r)

# ------------------ MAIN FLOW ------------------
def main():
    # 0) Export (once) — comment these two lines out if you already have local stacks
    export_ic_local_geedim(L3_BAIXO_D , BAIXO , "Baixo" , "BAIXO",  include_label=True)
    export_ic_local_geedim(L3_LAMEGO_D, LAMEGO, "Lamego", "LAMEGO", include_label=True)
    print("All local exports done.")

    # 1) files
    train_files, test_files = discover_files()

    # 2) features
    feature_names, lc_values = discover_predictor_names(
        train_files, do_one_hot=DO_ONE_HOT_LC, use_day=USE_DAY_OF_YEAR,
        max_scan=MAX_SCAN_FILES, pct_required=PCT_REQUIRED
    )
    print("[FEATURES RAW]", feature_names)
    if lc_values:
        print("[LC CODES]", lc_values)
        # expand LC into LC_<code>
        feature_names_expanded = []
        for f in feature_names:
            if f == "LC":
                for v in lc_values:
                    feature_names_expanded.append(f"LC_{v}")
            else:
                feature_names_expanded.append(f)
        feature_names = feature_names_expanded
    print("[FEATURES FINAL]", feature_names)

    # 3) matrix
    X_train, y_train = build_training_matrix(train_files, feature_names,
                                             per_file_samples=PER_FILE_SAMPLES,
                                             lc_values=lc_values)

    # 4) train RF
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    print("[RF] OOB score (if enabled):", getattr(rf, "oob_score_", None))

    # 5) feature importances
    imp = getattr(rf, "feature_importances_", None)
    if imp is not None:
        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
        print("[RF] Saved feature_importances.csv")

    # 6) save model + features
    dump({"rf": rf, "features": feature_names}, os.path.join(OUT_DIR, "rf_model.joblib"))
    with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
        json.dump({"features": feature_names, "lc_values": lc_values}, f, indent=2)
    print("[RF] Model saved.")

    # 7) predict test files
    records = []
    Path(VIS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PRED_DIR).mkdir(parents=True, exist_ok=True)

    for fp in test_files:
        base = os.path.basename(fp)
        tag  = os.path.splitext(base)[0]

        pred_tmp = os.path.join(PRED_DIR, f"{tag}_pred.tif")
        pred_out = os.path.join(PRED_DIR, f"{tag}_pred_clip.tif")
        lab_out  = os.path.join(PRED_DIR, f"{tag}_label_clip.tif")

        predict_stack_blockwise(fp, rf, feature_names, pred_tmp)
        clip_to_label_mask_pair(pred_tmp, fp, pred_out, lab_out)
        vis = visualise_minmax_robust(pred_out, lab_out, VIS_DIR, tag=tag, title_prefix="Region")
        m   = metrics_from_pair(pred_out, lab_out)

        row = dict(tag=tag, **m, **vis)
        print("[METRICS]", row)
        records.append(row)

    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print("[DONE] Metrics saved to metrics.csv")

if __name__ == "__main__":
    main()
