# -*- coding: utf-8 -*-
"""
WaPOR 20 m Downscaling — Local (Filter + Forward Selection + Tuning)

Pipeline:
  1) (Optional) Export dekadal stacks locally with geedim
  2) Discover features from exported stacks
  3) Mutual Information filter → top-K
  4) Greedy forward selection with GroupKFold-by-file (R²)
  5) (Optional) RandomizedSearchCV tuning
  6) Predict test stacks, clip to label mask, save PNGs and metrics

Auth:
  export EE_SERVICE_ACCOUNT_FILE=/path/to/your-service-account.json
"""

import os, re, json, glob, argparse, datetime as dt
from pathlib import Path
from collections import Counter

# Quiet some geedim shutdown warnings on certain Python builds
os.environ.setdefault("GD_DISABLE_ASYNC", "1")

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from joblib import dump
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

import ee
from google.oauth2 import service_account
import geedim  # registers .gd on ee.Image / ee.ImageCollection

# ================= AUTH =================
SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive",
]

def ee_init():
    sa_file = os.getenv("EE_SERVICE_ACCOUNT_FILE")
    if not sa_file or not os.path.isfile(sa_file):
        raise FileNotFoundError(
            "EE_SERVICE_ACCOUNT_FILE is not set or points to a missing file. "
            "Set it to your service-account JSON path."
        )
    credentials = service_account.Credentials.from_service_account_file(sa_file, scopes=SCOPES)
    ee.Initialize(credentials)

# ================= CONFIG =================
# AOIs & label collections
BAIXO_AOI  = "projects/tethys-app-1/assets/baixo"
LAMEGO_AOI = "projects/tethys-app-1/assets/lamego"

L3_BAIXO_D_COLL  = "projects/tethys-app-1/assets/WaPOR_L3_20m_D_BAIXO"
L3_LAMEGO_D_COLL = "projects/tethys-app-1/assets/WaPOR_L3_20m_D_LAMEGO"
LABEL_BAND       = "b1"  # adjust if your label differs

# Date windows (used for local train/test discovery only)
DEFAULT_TRAIN_START = "2018-01-01"; DEFAULT_TRAIN_END = "2024-12-01"
DEFAULT_TEST_START  = "2019-01-01"; DEFAULT_TEST_END  = "2022-12-21"

# Local output
OUTPUT_ROOT = Path("wapor_20m_local"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_DIR = str(OUTPUT_ROOT)  # BAIXO/ and LAMEGO/ subfolders expected
OUT_DIR  = OUTPUT_ROOT / "outputs_rf20_tuned"; OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = OUT_DIR / "preds"; PRED_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR  = OUT_DIR / "viz";   VIS_DIR.mkdir(parents=True, exist_ok=True)

# Export settings
EXPORT_SCALE       = 20
EXPORT_DTYPE       = "float32"
EXPORT_NODATA      = -9999
GEEDIM_TILE_MB     = 16  # must be < 32
GEEDIM_MAX_REQUEST = 16
MIN_VALID_BYTES    = 1024  # file size threshold to consider a file valid

# Sentinel-1 & texture config
S1_VALUES_ARE_DB = True
S1_SMOOTH_RADIUS = 1
TEXTURE_SIZE     = 3
DB_MIN, DB_MAX   = -25.0, 5.0

GLCM_MAP = {
    "contrast":      "contrast",
    "entropy":       "ent",
    "homogeneity":   "idm",
    "dissimilarity": "diss",
}

CHIRPS_LAG_DAYS = 10

# Tuned-flow discovery / sampling
LABEL_CANDIDATES  = ("b1","ETa20","ETa20m","AETI20","L3","LABEL","label")
USE_DAY_OF_YEAR   = True
MAX_SCAN_FILES    = 200
PCT_REQUIRED      = 0.90
RANDOM_STATE      = 7
NODATA_OUT        = -9999.0
CHUNK_PRED        = 1024
np.random.seed(RANDOM_STATE)

# RF base params (used if --tune not set; tuning searches around these)
RF_BASE_PARAMS = dict(
    n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    bootstrap=True, max_samples=0.8, oob_score=True, warm_start=False
)

# ========= Late-bound EE sources (bound after ee_init) =========
S2_SR = None
L1_AETI_D = None
DEM = None
SLOPE = None
LANDCOVER = None
S1_GRD = None
CHIRPS_DAILY = None
BAIXO = None
LAMEGO = None
L3_BAIXO_D = None
L3_LAMEGO_D = None

def bind_ee_sources():
    global S2_SR, L1_AETI_D, DEM, SLOPE, LANDCOVER, S1_GRD, CHIRPS_DAILY
    global BAIXO, LAMEGO, L3_BAIXO_D, L3_LAMEGO_D
    S2_SR        = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    L1_AETI_D    = ee.ImageCollection("FAO/WAPOR/3/L1_AETI_D")
    DEM          = ee.Image("USGS/SRTMGL1_003").rename("DEM")
    SLOPE        = ee.Terrain.slope(DEM).rename("Slope")
    LANDCOVER    = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("LC")
    S1_GRD       = (ee.ImageCollection("COPERNICUS/S1_GRD")
                     .filter(ee.Filter.eq("instrumentMode", "IW"))
                     .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
                     .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")))
    CHIRPS_DAILY = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    BAIXO  = ee.FeatureCollection(BAIXO_AOI)
    LAMEGO = ee.FeatureCollection(LAMEGO_AOI)
    L3_BAIXO_D  = ee.ImageCollection(L3_BAIXO_D_COLL).filterBounds(BAIXO)
    L3_LAMEGO_D = ee.ImageCollection(L3_LAMEGO_D_COLL).filterBounds(LAMEGO)

# ================= EE Helpers =================
def end_of_dekad_from_label_start(date):
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
    def q8(img, name_u8):
        return (img.unitScale(db_min, db_max).multiply(255).clamp(0, 255).toUint8().rename(name_u8))
    vv_u8 = q8(vvdb, "S1_VV_u8"); vh_u8 = q8(vhdb, "S1_VH_u8")
    vv_all = vv_u8.glcmTexture(size); vh_all = vh_u8.glcmTexture(size)
    vv_list = [vv_all.select(f"S1_VV_u8_{suf}").rename(f"S1_VV_{human}") for human, suf in GLCM_MAP.items()]
    vh_list = [vh_all.select(f"S1_VH_u8_{suf}").rename(f"S1_VH_{human}") for human, suf in GLCM_MAP.items()]
    return ee.Image.cat(vv_list + vh_list)

def s1_dekad_safe(start, end, region):
    coll = S1_GRD.filterBounds(region).filterDate(start, end)
    def _make():
        vv = coll.select("VV").median(); vh = coll.select("VH").median()
        vvdb = vv if S1_VALUES_ARE_DB else to_db(vv)
        vhdb = vh if S1_VALUES_ARE_DB else to_db(vh)
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
    coll = CHIRPS_DAILY.filterDate(start, end).select("precipitation")
    img  = coll.sum().rename(band_name)
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), img, ee.Image(0).updateMask(ee.Image(0)).rename(band_name)))

def chirps_dekad_pair(start, end):
    cur  = chirps_sum(start, end, "CHIRPS10d")
    prev_start = ee.Date(start).advance(-CHIRPS_LAG_DAYS, "day")
    prev_end   = ee.Date(start)
    lag  = chirps_sum(prev_start, prev_end, "CHIRPS10d_lag")
    return cur.addBands(lag)

def build_stack_for_label(label_img, region_fc, include_label=True):
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

# ============== Export with geedim (skip/overwrite) ==============
def export_ic_local_geedim(label_ic, region_fc, region_name, subfolder,
                           include_label=True, scale=EXPORT_SCALE,
                           dtype=EXPORT_DTYPE, nodata_val=EXPORT_NODATA,
                           max_tile_size_mb=GEEDIM_TILE_MB, max_requests=GEEDIM_MAX_REQUEST,
                           overwrite=False):
    # robustly handle empty collections
    try:
        n = int(ee.Number(label_ic.size()).getInfo())
    except Exception as e:
        print(f"[SKIP] {region_name}: cannot determine collection size ({e}).")
        return
    if n == 0:
        print(f"[SKIP] {region_name}: label collection is empty for this window.")
        return

    ic   = label_ic.sort("system:time_start")
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

        if os.path.exists(fpath) and not overwrite:
            try:
                if os.path.getsize(fpath) > MIN_VALID_BYTES:
                    print(f"[SKIP EXISTING] {fpath}")
                    continue
                else:
                    print(f"[RE-DOWNLOAD SMALL FILE] {fpath} ({os.path.getsize(fpath)} bytes)")
            except OSError:
                pass

        prepared = (stack.clip(geom).gd.prepareForExport(region=geom, scale=scale, dtype=dtype))
        try:
            prepared.gd.toGeoTIFF(
                fpath,
                overwrite=True if overwrite else False,
                nodata=nodata_val,
                max_tile_size=max_tile_size_mb,
                max_requests=max_requests
            )
            print("Saved:", fpath)
        except Exception as e:
            print(f"[WARN] Failed {fname}: {e}")

# ================= Local discovery & IO =================
def parse_any_date(text):
    # FIX: use raw strings; earlier code had escaped backslashes and failed
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', text) or re.search(r'(\d{4})(\d{2})(\d{2})', text)
    if not m: return None, None
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

def discover_files(data_dir: str, train_start: str, train_end: str, test_start: str, test_end: str):
    baixo_dir  = os.path.join(data_dir, "BAIXO")
    lamego_dir = os.path.join(data_dir, "LAMEGO")

    if os.path.isdir(baixo_dir) and os.path.isdir(lamego_dir):
        train = sorted(glob.glob(os.path.join(baixo_dir, "*stack*.tif")))
        test  = sorted(glob.glob(os.path.join(lamego_dir, "*stack*.tif")))
        print(f"[DISCOVERY] BAIXO={len(train)} | LAMEGO={len(test)}")
        return train, test

    files = sorted(glob.glob(os.path.join(data_dir, "*stack*.tif")))
    print(f"[DISCOVERY] DATA_DIR={data_dir}, found {len(files)} stack files")
    if not files:
        return [], []
    rows = []
    for fp in files:
        dstr, doy = parse_any_date(os.path.basename(fp))
        rows.append((fp, dstr, doy))
    df = pd.DataFrame(rows, columns=["fp","date","doy"]).dropna(subset=["date"]).copy()
    if df.empty:
        split = max(1, int(0.7 * len(files)))
        return files[:split], files[split:]
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")
    train = [r.fp for r in df.itertuples(index=False) if in_range(r.date, train_start, train_end, False)]
    test  = [r.fp for r in df.itertuples(index=False) if in_range(r.date,  test_start,  test_end,  True)]
    if len(train) == 0:
        split = max(1, int(0.7 * len(df)))
        train = df["fp"].iloc[:split].tolist(); test = df["fp"].iloc[split:].tolist()
    if len(test) == 0:
        rest = [f for f in files if f not in set(train)]
        test = rest
    print(f"[DISCOVERY] train: {len(train)} | test: {len(test)}")
    return train, test

def band_map(ds):
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
        return ds.read(ds.count).astype(np.float32)
    return ds.read(bm[nm]+1).astype(np.float32)

# ================= Feature discovery (numeric LC) =================
def discover_predictor_names(train_files, use_day=True, max_scan=MAX_SCAN_FILES, pct_required=PCT_REQUIRED):
    counts, total = Counter(), 0
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
    keep = sorted([nm for nm, c in counts.items() if c / total >= pct_required])
    if use_day:
        keep += ["DAY_SIN","DAY_COS"]
    return keep

# ================= Sampling / matrices (numeric LC) =================
def sample_from_stack_numericLC(path, features, n=1000):
    with rasterio.open(path) as ds:
        bm = band_map(ds)
        label = read_label_band(ds, bm)
        nodata = ds.nodata
        mask = np.isfinite(label)
        if nodata is not None and np.isfinite(nodata):
            mask &= (label != nodata)

        needed = set()
        for f in features:
            if f in ("DAY_SIN","DAY_COS"): continue
            needed.add(f)

        raw = {}
        for name in needed:
            if name not in bm: return None, None
            raw[name] = ds.read(bm[name]+1).astype(np.float32)

    base = os.path.basename(path)
    date_str, doy = parse_any_date(base)
    angle   = 2.0 * np.pi * ((doy or 1) / 365.0)
    day_sin = np.float32(np.sin(angle))
    day_cos = np.float32(np.cos(angle))

    for f in features:
        if f in ("DAY_SIN","DAY_COS"): continue
        a = raw.get(f)
        if a is None: return None, None
        mask &= np.isfinite(a)

    ys, xs = np.where(mask)
    if ys.size == 0: return None, None
    take = min(n, ys.size)
    sel = np.random.choice(ys.size, size=take, replace=False)
    ys, xs = ys[sel], xs[sel]

    cols = []
    for f in features:
        if f == "DAY_SIN":
            cols.append(np.full_like(ys, day_sin, dtype=np.float32))
        elif f == "DAY_COS":
            cols.append(np.full_like(ys, day_cos, dtype=np.float32))
        else:
            cols.append(raw[f][ys, xs])

    X = np.stack(cols, axis=1).astype(np.float32)
    y = label[ys, xs].astype(np.float32)
    return X, y

def build_matrix(files, features, per_file_samples=800):
    Xs, ys, gs = [], [], []
    for fi, fp in enumerate(files):
        Xi, yi = sample_from_stack_numericLC(fp, features, n=per_file_samples)
        if Xi is None: continue
        Xs.append(Xi); ys.append(yi); gs.append(np.full((Xi.shape[0],), fi, dtype=np.int32))
    if not Xs:
        raise RuntimeError("No samples assembled for features: %s" % features)
    X = np.vstack(Xs); y = np.concatenate(ys); groups = np.concatenate(gs)
    print(f"[MATRIX] X={X.shape} from files={len(Xs)}")
    return X, y, groups

# ================= Filter & Forward Selection =================
def mi_filter_topk(train_files, base_pool, per_file_samples=800, top_k=10, random_state=RANDOM_STATE):
    take = min(len(train_files), max(4, len(train_files)//3))
    subset = train_files[:take]
    X, y, _ = build_matrix(subset, base_pool, per_file_samples=per_file_samples)
    valid_cols = np.all(np.isfinite(X), axis=0)
    Xv = X[:, valid_cols]
    feats = [f for (f, ok) in zip(base_pool, valid_cols) if ok]
    mi = mutual_info_regression(Xv, y, random_state=random_state)
    order = np.argsort(mi)[::-1]
    selected = [feats[i] for i in order[:min(top_k, len(feats))]]
    scores = {feats[i]: float(mi[i]) for i in order}
    print("[FILTER] Selected (MI top K):", selected)
    return selected, scores

def forward_select(train_files, pool, per_file_samples=800, cv_splits=3,
                   min_improve=1e-4, max_steps=12, random_state=RANDOM_STATE):
    current = []
    best_score = -np.inf
    steps = 0

    while pool and steps < max_steps:
        steps += 1
        candidates = []
        for f in pool:
            feats = current + [f]
            X, y, groups = build_matrix(train_files, feats, per_file_samples=per_file_samples)
            n_groups = len(np.unique(groups))

            rf = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=random_state, n_jobs=-1)

            if n_groups >= 2:
                n_splits = min(cv_splits, n_groups)
                gkf = GroupKFold(n_splits=n_splits)
                scores = []
                for tr, va in gkf.split(X, y, groups):
                    rf.fit(X[tr], y[tr])
                    p = rf.predict(X[va])
                    scores.append(r2_score(y[va], p))
                score = float(np.mean(scores))
            else:
                idx = np.arange(X.shape[0])
                tr, va = train_test_split(idx, test_size=0.2, random_state=random_state)
                rf.fit(X[tr], y[tr])
                p = rf.predict(X[va])
                score = float(r2_score(y[va], p))

            candidates.append((f, score))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_f, best_f_score = candidates[0]
        print(f"[FWD] Step {steps}: try best add '{best_f}' → R²={best_f_score:.4f}")

        if best_f_score > (best_score + min_improve):
            best_score = best_f_score
            current.append(best_f)
            pool = [x for x in pool if x != best_f]
        else:
            print("[FWD] Improvement below threshold, stopping.")
            break

    print("[FWD] Selected:", current, "R²≈", round(best_score, 4))
    return current, best_score

# ================= Tuning =================
def tune_rf(X, y, random_state=RANDOM_STATE, n_iter=30, cv_splits=5):
    rf = RandomForestRegressor(**RF_BASE_PARAMS)
    param_dist = dict(
        n_estimators=[300, 500, 700, 900, 1200],
        max_depth=[None, 10, 20, 30, 40],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["sqrt", 0.4, 0.6, 0.8],
        bootstrap=[True],
        max_samples=[None, 0.6, 0.8, 0.9],
    )
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        scoring="neg_mean_squared_error", cv=cv_splits,
        verbose=1, n_jobs=-1, random_state=random_state, refit=True
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

# ================= Prediction / Post-process =================
def predict_stack_blockwise(fp_in, rf, feature_names, fp_out, chunk=CHUNK_PRED, nodata=NODATA_OUT):
    with rasterio.open(fp_in) as src:
        bm = band_map(src)
        H, W = src.height, src.width
        prof = src.profile.copy()
        out_prof = prof.copy()
        out_prof.update(count=1, dtype="float32", nodata=nodata, compress="deflate", predictor=3, tiled=True)

        base = os.path.basename(fp_in)
        date_str, doy = parse_any_date(base)
        angle   = 2.0 * np.pi * ((doy or 1) / 365.0)
        day_sin = np.float32(np.sin(angle))
        day_cos = np.float32(np.cos(angle))

        with rasterio.open(fp_out, "w", **out_prof) as dst:
            for y0 in range(0, H, chunk):
                for x0 in range(0, W, chunk):
                    h0 = min(chunk, H - y0); w0 = min(chunk, W - x0)
                    needed = set(f for f in feature_names if f not in ("DAY_SIN","DAY_COS"))
                    raw = {}
                    missing = False
                    for name in needed:
                        if name not in bm: missing = True; break
                        raw[name] = src.read(bm[name]+1, window=Window(x0, y0, w0, h0)).astype(np.float32)
                    if missing:
                        dst.write(np.full((h0,w0), nodata, np.float32), 1, window=Window(x0, y0, w0, h0))
                        continue

                    feats = []
                    for f in feature_names:
                        if f == "DAY_SIN":
                            feats.append(np.full((h0,w0), day_sin, dtype=np.float32))
                        elif f == "DAY_COS":
                            feats.append(np.full((h0,w0), day_cos, dtype=np.float32))
                        else:
                            feats.append(raw[f])

                    block = np.stack(feats, axis=0)
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

# ================= CLI entry =================
def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="Enable RF hyperparameter tuning (RandomizedSearchCV)")
    ap.add_argument("--skip-export", action="store_true", help="Skip geedim exports if local stacks already exist")
    ap.add_argument("--overwrite", action="store_true", help="Force re-download even if file exists")

    # Filter & forward params
    ap.add_argument("--filter-top-k", type=int, default=10, help="MI filter shortlist size (default: 10)")
    ap.add_argument("--filter-samples", type=int, default=800, help="Per-file samples in MI filter stage")
    ap.add_argument("--forward-samples", type=int, default=800, help="Per-file samples in forward-selection stage")
    ap.add_argument("--forward-cv", type=int, default=3, help="GroupKFold splits (caps to #files)")
    ap.add_argument("--forward-max-steps", type=int, default=12, help="Max steps for greedy forward selection")
    ap.add_argument("--forward-min-improve", type=float, default=1e-4, help="Min R² improvement to continue")

    # Tuning params
    ap.add_argument("--tune-iter", type=int, default=30, help="RandomizedSearchCV iterations")
    ap.add_argument("--tune-cv", type=int, default=5, help="CV splits during tuning")

    # Date overrides for local discovery
    ap.add_argument("--train-start", type=str, default=DEFAULT_TRAIN_START)
    ap.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    ap.add_argument("--test-start", type=str, default=DEFAULT_TEST_START)
    ap.add_argument("--test-end", type=str, default=DEFAULT_TEST_END)

    args = ap.parse_args()

    # EE init & bind sources AFTER initialization
    ee_init()
    bind_ee_sources()

    # 0) Export
    if args.skip_export:
        print("[EXPORT] Skipped by --skip-export")
    else:
        export_ic_local_geedim(L3_BAIXO_D , BAIXO , "Baixo" , "BAIXO",  include_label=True, overwrite=args.overwrite)
        export_ic_local_geedim(L3_LAMEGO_D, LAMEGO, "Lamego", "LAMEGO", include_label=True, overwrite=args.overwrite)
        print("All local exports done.")

    # 1) Discover files
    train_files, test_files = discover_files(DATA_DIR, args.train_start, args.train_end, args.test_start, args.test_end)
    if not train_files or not test_files:
        raise RuntimeError("No training or test files discovered. Ensure exports exist or adjust date windows.")

    # 2) Base pool (numeric LC; keep DOY)
    base_feats = discover_predictor_names(train_files, use_day=USE_DAY_OF_YEAR,
                                          max_scan=MAX_SCAN_FILES, pct_required=PCT_REQUIRED)

    # 3) MI filter
    shortlist, mi_scores = mi_filter_topk(train_files, base_feats, per_file_samples=args.filter_samples, top_k=args.filter_top_k)
    pd.DataFrame([{"feature": k, "mi": v} for k, v in mi_scores.items()]).to_csv(OUT_DIR / "mi_scores.csv", index=False)

    # 4) Forward selection
    selected, best_r2 = forward_select(train_files, shortlist, per_file_samples=args.forward_samples,
                                       cv_splits=args.forward_cv, min_improve=args.forward_min_improve,
                                       max_steps=args.forward_max_steps)
    with open(OUT_DIR / "forward_selected.json", "w") as f:
        json.dump({"selected": selected, "best_r2_cv": best_r2}, f, indent=2)

    # 5) Final train matrix
    X_train, y_train, _ = build_matrix(train_files, selected, per_file_samples=args.forward_samples)

    # 6) Fit or tune
    if args.tune:
        rf, best_params = tune_rf(X_train, y_train, n_iter=args.tune_iter, cv_splits=args.tune_cv)
        with open(OUT_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        rf = RandomForestRegressor(**RF_BASE_PARAMS)
        rf.fit(X_train, y_train)

    dump({"rf": rf, "features": selected}, OUT_DIR / "rf_model_tuned.joblib")
    with open(OUT_DIR / "features_tuned.json", "w") as f:
        json.dump({"features": selected}, f, indent=2)

    # 7) Predict & evaluate
    records = []
    for fp in test_files:
        base = os.path.basename(fp); tag = os.path.splitext(base)[0]
        pred_tmp = str(PRED_DIR / f"{tag}_pred.tif")
        pred_out = str(PRED_DIR / f"{tag}_pred_clip.tif")
        lab_out  = str(PRED_DIR / f"{tag}_label_clip.tif")

        predict_stack_blockwise(fp, rf, selected, pred_tmp)
        clip_to_label_mask_pair(pred_tmp, fp, pred_out, lab_out)
        vis = visualise_minmax_robust(pred_out, lab_out, str(VIS_DIR), tag=tag, title_prefix="Region")
        m   = metrics_from_pair(pred_out, lab_out)
        row = dict(tag=tag, **m, **vis)
        print("[METRICS]", row)
        records.append(row)

    pd.DataFrame(records).to_csv(OUT_DIR / "metrics_tuned.csv", index=False)
    print("[DONE] metrics_tuned.csv")

    # best-effort geedim async runner close
    try:
        import geedim.utils as gd_utils
        gd_utils.AsyncRunner()._close()
    except Exception:
        pass

if __name__ == "__main__":
    main_cli()
