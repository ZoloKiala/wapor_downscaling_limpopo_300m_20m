#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a trained 20 m RF downscaling model to ANY AOI for ONE dekad date.
- Downloads a 20 m predictor stack (S2, NDVI, S1 dB+GLCM, CHIRPS & lag, DEM, Slope, LC, ETa300m)
  for the given AOI/date via Earth Engine + geedim.
- Loads your RF model + feature list, predicts the 20 m ETa.
- Clips to the WaPOR ETa300m footprint and writes a "Predicted vs WaPOR ETa" PNG.
- Service account creds are OPTIONAL & hidden (env/cached auth used if omitted).

CLI:
  python predict_any_aoi.py \
    --aoi "bbox:16.2,-29.1,17.7,-28.4" \
    --date 2020-01-01 \
    --model wapor_20m_local/outputs_rf20_auto/rf_model.joblib \
    --features wapor_20m_local/outputs_rf20_auto/features.json \
    --out_root wapor_20m_any_aoi

Auth options (pick one; do NOT pass JSON on CLI if you want it hidden):
  export EE_SERVICE_ACCOUNT_FILE=/secure/sa.json
  # or: export GOOGLE_APPLICATION_CREDENTIALS=/secure/sa.json
  # or: export EE_SERVICE_ACCOUNT_JSON="$(cat /secure/sa.json)"
  # or: export EE_SERVICE_ACCOUNT_JSON_B64="$(base64 -w0 /secure/sa.json)"
  # or: run once `earthengine authenticate` (cached OAuth)
"""

import os, re, json, argparse
from pathlib import Path

# Headless plotting backend for servers/WSL
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from joblib import load

import ee
from google.oauth2 import service_account
import geedim  # registers .gd

# ------------------------------------------------------------
# Optional .env loader (no extra deps)
# ------------------------------------------------------------
def _load_dotenv_if_present(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ------------------------------------------------------------
# EE dataset IDs (lazy creation after auth)
# ------------------------------------------------------------
S2_SR_ID         = "COPERNICUS/S2_SR_HARMONIZED"
WAPOR_L1_AETI_ID = "FAO/WAPOR/3/L1_AETI_D"
SRTM_ID          = "USGS/SRTMGL1_003"
WORLDCOVER_ID    = "ESA/WorldCover/v200"
S1_GRD_ID        = "COPERNICUS/S1_GRD"
CHIRPS_DAILY_ID  = "UCSB-CHG/CHIRPS/DAILY"

# Texture config (GLCM on 8-bit quantized dB)
TEXTURE_SIZE = 3
DB_MIN, DB_MAX = -25.0, 5.0
GLCM_MAP = {"contrast":"contrast","entropy":"ent","homogeneity":"idm","dissimilarity":"diss"}

# Export params
EXPORT_SCALE = 20
EXPORT_DTYPE = "float32"
EXPORT_NODATA = -9999.0
GEEDIM_TILE_MB = 16
GEEDIM_MAX_REQUESTS = 16

# Prediction params
CHUNK_PRED = 1024
NODATA_OUT = -9999.0

# ------------------------------------------------------------
# Auth (hidden / optional)
# ------------------------------------------------------------
def _ee_init_auto(sa_json: str | None = None, verbose: bool = True):
    """
    Initialize Earth Engine without exposing JSON on CLI.
    Order:
      1) --sa_json (explicit path)
      2) EE_SERVICE_ACCOUNT_FILE / GOOGLE_APPLICATION_CREDENTIALS (path)
      3) EE_SERVICE_ACCOUNT_JSON (inline JSON)
      4) EE_SERVICE_ACCOUNT_JSON_B64 (base64 inline)
      5) cached OAuth (ee.Initialize())
      6) Google ADC (google.auth.default)
    """
    scopes = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/drive",
    ]
    def _ok(msg):
        if verbose:
            print(f"[EE AUTH] {msg}")

    # 1) explicit file
    if sa_json:
        cred = service_account.Credentials.from_service_account_file(sa_json, scopes=scopes)
        ee.Initialize(cred)
        _ok(f"Initialized from --sa_json file: {sa_json}")
        return

    # 2) env file path
    for var in ("EE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"):
        p = os.getenv(var, "")
        if p and os.path.exists(p):
            cred = service_account.Credentials.from_service_account_file(p, scopes=scopes)
            ee.Initialize(cred)
            _ok(f"Initialized from {var}={p}")
            return

    # 3) env inline JSON
    inline = os.getenv("EE_SERVICE_ACCOUNT_JSON")
    if inline:
        info = json.loads(inline)
        cred = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        ee.Initialize(cred)
        _ok("Initialized from EE_SERVICE_ACCOUNT_JSON (inline JSON).")
        return

    # 4) env base64 inline JSON
    b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        import base64
        info = json.loads(base64.b64decode(b64).decode("utf-8"))
        cred = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        ee.Initialize(cred)
        _ok("Initialized from EE_SERVICE_ACCOUNT_JSON_B64 (base64 inline).")
        return

    # 5) cached OAuth
    try:
        ee.Initialize()
        _ok("Initialized from cached OAuth credentials (ee.Initialize()).")
        return
    except Exception:
        pass

    # 6) Google ADC
    import google.auth
    creds, proj = google.auth.default(scopes=scopes)
    ee.Initialize(creds)
    _ok(f"Initialized from Google ADC (project={proj}).")

# ------------------------------------------------------------
# Helpers: AOI / Dates
# ------------------------------------------------------------
def _fc_from_any_aoi(aoi: str) -> ee.FeatureCollection:
    """
    AOI formats:
      - EE asset id: 'projects/...', 'users/...'
      - bbox: 'bbox:minlon,minlat,maxlon,maxlat'
      - local GeoJSON: '/path/to/aoi.geojson'
    """
    if aoi.startswith(("projects/","users/")):
        return ee.FeatureCollection(aoi)
    if aoi.startswith("bbox:"):
        parts = aoi.split("bbox:",1)[1].split(",")
        xmin, ymin, xmax, ymax = map(float, parts)
        geom = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], geodesic=False)
        return ee.FeatureCollection(ee.Feature(geom))
    if aoi.lower().endswith((".geojson",".json")) and os.path.exists(aoi):
        with open(aoi, "r") as f:
            gj = json.load(f)
        t = gj.get("type","")
        if t == "FeatureCollection":
            geom = gj["features"][0]["geometry"]
        elif t == "Feature":
            geom = gj["geometry"]
        else:
            geom = gj  # raw Geometry
        return ee.FeatureCollection(ee.Feature(ee.Geometry(geom)))
    raise ValueError("Unsupported AOI. Use EE asset id, bbox:..., or a .geojson path.")

def _aoi_name(aoi: str) -> str:
    if aoi.startswith("bbox:"):
        return "aoi_" + aoi.split("bbox:",1)[1].replace(",","_").replace(" ","")
    base = os.path.basename(aoi.rstrip("/"))
    return re.sub(r"[^A-Za-z0-9_\-\.]+","_", base or "aoi")

def _end_of_dekad(start_iso: str) -> ee.Date:
    d = ee.Number(ee.Date(start_iso).get('day'))
    is_third = d.gte(21)
    return ee.Date(ee.Algorithms.If(is_third, ee.Date(start_iso).advance(1,'month'),
                                    ee.Date(start_iso).advance(10,'day')))

# ------------------------------------------------------------
# Lazy EE sources and stack builders (constructed AFTER auth)
# ------------------------------------------------------------
def _s2_dekad_safe(start, end, region_fc):
    win   = ee.ImageCollection(S2_SR_ID).filterBounds(region_fc).filterDate(start, end)
    bnames = ["B2","B3","B4","B8","B11","B12"]
    win70 = win.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
    img70 = win70.median().select(bnames)
    imgAny= win.median().select(bnames)
    masked= ee.Image.constant([0]*len(bnames)).rename(bnames).updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(win70.size().gt(0), img70,
           ee.Algorithms.If(win.size().gt(0), imgAny, masked)))

def _l1_dekad_safe(start, end, region_fc):
    coll = ee.ImageCollection(WAPOR_L1_AETI_ID).filterBounds(region_fc).filterDate(start, end)
    img  = coll.mean().select([0]).rename("ETa300m")
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), img,
           ee.Image(0).updateMask(ee.Image(0)).rename("ETa300m")))

def _s1_textures_from_db(vvdb, vhdb, size=TEXTURE_SIZE, db_min=DB_MIN, db_max=DB_MAX):
    def q8(img, name_u8):
        return (img.unitScale(db_min, db_max).multiply(255).clamp(0,255).toUint8().rename(name_u8))
    vv_u8 = q8(vvdb, "S1_VV_u8")
    vh_u8 = q8(vhdb, "S1_VH_u8")
    vv_all = vv_u8.glcmTexture(size)
    vh_all = vh_u8.glcmTexture(size)
    vv_list = [vv_all.select(f"S1_VV_u8_{suf}").rename(f"S1_VV_{k}") for k, suf in GLCM_MAP.items()]
    vh_list = [vh_all.select(f"S1_VH_u8_{suf}").rename(f"S1_VH_{k}") for k, suf in GLCM_MAP.items()]
    return ee.Image.cat(vv_list + vh_list)

def _s1_dekad_safe(start, end, region_fc):
    coll = (ee.ImageCollection(S1_GRD_ID)
            .filterBounds(region_fc).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode","IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")))
    def _make():
        vvdb = coll.select("VV").median()
        vhdb = coll.select("VH").median()
        vv_s = vvdb.focal_mean(radius=1, units="pixels").rename("S1_VV_dB")
        vh_s = vhdb.focal_mean(radius=1, units="pixels").rename("S1_VH_dB")
        diff = vvdb.subtract(vhdb).focal_mean(radius=1, units="pixels").rename("S1_VVminusVH_dB")
        tex  = _s1_textures_from_db(vvdb, vhdb)
        return ee.Image.cat([vv_s, vh_s, diff, tex])
    placeholders = [
        "S1_VV_dB","S1_VH_dB","S1_VVminusVH_dB",
        "S1_VV_contrast","S1_VV_entropy","S1_VV_homogeneity","S1_VV_dissimilarity",
        "S1_VH_contrast","S1_VH_entropy","S1_VH_homogeneity","S1_VH_dissimilarity",
    ]
    masked = ee.Image.constant([0]*len(placeholders)).rename(placeholders).updateMask(ee.Image(0))
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), _make(), masked))

def _chirps_sum(start, end, name):
    coll = ee.ImageCollection(CHIRPS_DAILY_ID).filterDate(start, end).select("precipitation")
    img  = coll.sum().rename(name)
    return ee.Image(ee.Algorithms.If(coll.size().gt(0), img,
                                     ee.Image(0).updateMask(ee.Image(0)).rename(name)))

def _chirps_pair(start, end):
    cur = _chirps_sum(start, end, "CHIRPS10d")
    lag = _chirps_sum(ee.Date(start).advance(-10, "day"), start, "CHIRPS10d_lag")
    return cur.addBands(lag)

def _build_stack_for_date(date_iso: str, region_fc: ee.FeatureCollection):
    start = ee.Date(date_iso); end = _end_of_dekad(date_iso)
    s2    = _s2_dekad_safe(start, end, region_fc)
    ndvi  = s2.normalizedDifference(["B8","B4"]).rename("NDVI")
    l1    = _l1_dekad_safe(start, end, region_fc)
    s1    = _s1_dekad_safe(start, end, region_fc)
    rain2 = _chirps_pair(start, end)

    dem   = ee.Image(SRTM_ID).rename("DEM")
    slope = ee.Terrain.slope(dem).rename("Slope")
    lc    = ee.ImageCollection(WORLDCOVER_ID).first().select("Map").rename("LC")

    stack = (s2.addBands([ndvi, l1, dem, slope, lc, s1, rain2])
             .select([
                 "B4","B8","B11","NDVI","ETa300m",
                 "DEM","Slope","LC",
                 "S1_VV_dB","S1_VH_dB","S1_VVminusVH_dB",
                 "S1_VV_contrast","S1_VV_entropy","S1_VV_homogeneity","S1_VV_dissimilarity",
                 "S1_VH_contrast","S1_VH_entropy","S1_VH_homogeneity","S1_VH_dissimilarity",
                 "CHIRPS10d","CHIRPS10d_lag"
             ]))
    return stack

# ------------------------------------------------------------
# Geedim download
# ------------------------------------------------------------
def ensure_stack_download(date: str, aoi: str, out_root: str, sa_json: str | None = None) -> tuple[str,str]:
    _ee_init_auto(sa_json)
    fc = _fc_from_any_aoi(aoi)
    geom = fc.geometry()
    aoi_name = _aoi_name(aoi)
    out_dir  = os.path.join(out_root, aoi_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stack_fp = os.path.join(out_dir, f"{aoi_name}_{date}_stack.tif")
    if os.path.exists(stack_fp):
        print("[STACK] Reusing:", stack_fp)
        return stack_fp, aoi_name
    stack = _build_stack_for_date(date, fc)
    prepared = (stack.clip(geom)
                     .gd.prepareForExport(region=geom, scale=EXPORT_SCALE, dtype=EXPORT_DTYPE))
    prepared.gd.toGeoTIFF(stack_fp, overwrite=True, nodata=EXPORT_NODATA,
                          max_tile_size=GEEDIM_TILE_MB, max_requests=GEEDIM_MAX_REQUESTS)
    print("[STACK] Saved:", stack_fp)
    return stack_fp, aoi_name

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def _band_map(ds):
    desc = list(ds.descriptions or [])
    return {desc[i]: i for i in range(ds.count) if i < len(desc) and desc[i]}

def _one_hot_lc(lc_arr, code):
    lc_int = np.where(np.isfinite(lc_arr), lc_arr, -9999).astype(np.int32, copy=False)
    return (lc_int == int(code)).astype(np.float32)

def _day_sin_cos_from_date(date_iso):
    import datetime as dt
    y, m, d = map(int, date_iso.split("-"))
    doy = (dt.date(y, m, d) - dt.date(y, 1, 1)).days + 1
    ang = 2.0 * np.pi * (doy / 365.0)
    return np.float32(np.sin(ang)), np.float32(np.cos(ang))

def predict_stack_blockwise(fp_in, rf, features, fp_out, date_iso, chunk=CHUNK_PRED, nodata=NODATA_OUT):
    with rasterio.open(fp_in) as src:
        bm = _band_map(src)
        H, W = src.height, src.width
        prof = src.profile.copy()
        out_prof = prof.copy()
        out_prof.update(count=1, dtype="float32", nodata=nodata,
                        compress="deflate", predictor=3, tiled=True)
        day_sin, day_cos = _day_sin_cos_from_date(date_iso)

        with rasterio.open(fp_out, "w", **out_prof) as dst:
            for y0 in range(0, H, chunk):
                for x0 in range(0, W, chunk):
                    h0 = min(chunk, H-y0); w0 = min(chunk, W-x0)
                    need = set()
                    for f in features:
                        if f.startswith("LC_"): need.add("LC")
                        elif f not in ("DAY_SIN","DAY_COS"):
                            need.add(f)
                    raw = {}
                    ok = True
                    for f in need:
                        if f not in bm:
                            ok = False; break
                        raw[f] = src.read(bm[f]+1, window=Window(x0,y0,w0,h0)).astype(np.float32)
                    if not ok or not raw:
                        dst.write(np.full((h0,w0), nodata, np.float32), 1, window=Window(x0,y0,w0,h0))
                        continue

                    cols = []
                    for f in features:
                        if f == "DAY_SIN":
                            cols.append(np.full((h0,w0), day_sin, dtype=np.float32))
                        elif f == "DAY_COS":
                            cols.append(np.full((h0,w0), day_cos, dtype=np.float32))
                        elif f.startswith("LC_"):
                            code = int(f.split("_",1)[1])
                            a = raw.get("LC", None)
                            if a is None: ok=False; break
                            cols.append(_one_hot_lc(a, code))
                        else:
                            a = raw.get(f, None)
                            if a is None: ok=False; break
                            cols.append(a)
                    if not ok:
                        dst.write(np.full((h0,w0), nodata, np.float32), 1, window=Window(x0,y0,w0,h0))
                        continue

                    block = np.stack(cols, axis=0)
                    X = np.moveaxis(block, 0, -1).reshape(-1, len(features))
                    valid = np.isfinite(X).all(axis=1)
                    pred = np.full((X.shape[0],), np.nan, np.float32)
                    if valid.any():
                        pred[valid] = rf.predict(X[valid])
                    pred = pred.reshape(h0, w0)
                    dst.write(pred, 1, window=Window(x0,y0,w0,h0))
    return fp_out

# ------------------------------------------------------------
# Clip & Viz vs ETa300m
# ------------------------------------------------------------
def _read_eta300_from_stack(ds):
    bm = _band_map(ds)
    for k in bm:
        low = (k or "").lower()
        if ("eta" in low or "aeti" in low) and "300" in low:
            return ds.read(bm[k]+1).astype(np.float32)
    if ds.count >= 5:
        try:
            return ds.read(5).astype(np.float32)
        except Exception:
            pass
    return None

def clip_to_eta300_mask_pair(pred_tif, stack_tif, out_pred_tif, out_eta_tif, nodata=NODATA_OUT):
    with rasterio.open(stack_tif) as ds, rasterio.open(pred_tif) as dp:
        eta = _read_eta300_from_stack(ds)
        if eta is None:
            raise KeyError("ETa300m band not found in downloaded stack.")
        pred = dp.read(1).astype(np.float32)
        pred_transform = dp.transform
        prof = dp.profile.copy()
    m = np.isfinite(eta)
    pred[~m] = nodata
    ys, xs = np.where(m)
    if ys.size == 0:
        raise RuntimeError("Empty ETa300m mask.")
    ymin, ymax = ys.min(), ys.max()+1
    xmin, xmax = xs.min(), xs.max()+1
    win = Window(xmin, ymin, xmax-xmin, ymax-ymin)
    new_transform = rasterio.windows.transform(win, pred_transform)
    pred_c = pred[ymin:ymax, xmin:xmax]
    eta_c  = eta [ymin:ymax, xmin:xmax]
    prof.update(height=pred_c.shape[0], width=pred_c.shape[1],
                transform=new_transform, count=1, dtype="float32",
                nodata=nodata, compress="deflate", predictor=3, tiled=True)
    with rasterio.open(out_pred_tif, "w", **prof) as d1: d1.write(pred_c, 1)
    with rasterio.open(out_eta_tif,  "w", **prof) as d2: d2.write(eta_c,  1)

def visualise_minmax_robust(pred_clip, eta_clip, out_png, title="Predicted vs WaPOR ETa (mm/day)"):
    with rasterio.open(pred_clip) as dp, rasterio.open(eta_clip) as dn:
        p = dp.read(1).astype(np.float32); t = dn.read(1).astype(np.float32)
        ndp, ndt = dp.nodata, dn.nodata
    valid = np.isfinite(p) & np.isfinite(t)
    if ndp is not None: valid &= (p != ndp)
    if ndt is not None: valid &= (t != ndt)
    pm = np.ma.array(p, mask=~valid); tm = np.ma.array(t, mask=~valid)
    if pm.count()==0 or tm.count()==0:
        return None
    vmin = float(np.min([pm.min(), tm.min()]))
    vmax = float(np.max([pm.max(), tm.max()]))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin >= vmax:
        vmin, vmax = 0.0, 8.0
    cmap = plt.cm.viridis.copy(); cmap.set_bad("white", 0.0)
    fig = plt.figure(figsize=(12,5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 0.06])
    axL = fig.add_subplot(gs[0,0]); axR = fig.add_subplot(gs[0,1]); cax = fig.add_subplot(gs[1,:])
    axL.imshow(tm, cmap=cmap, vmin=vmin, vmax=vmax); axL.set_title("WaPOR ETa (native)"); axL.axis("off")
    imR = axR.imshow(pm, cmap=cmap, vmin=vmin, vmax=vmax); axR.set_title("Predicted 20 m"); axR.axis("off")
    cb = fig.colorbar(imR, cax=cax, orientation="horizontal"); cb.set_label("ETa (mm/day)")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    return out_png

# ------------------------------------------------------------
# Model I/O & metrics
# ------------------------------------------------------------
def _load_model_and_features(model_path, feats_path=None):
    obj = load(model_path)
    if isinstance(obj, dict):
        rf = obj.get("rf", None) or obj.get("model", None)
        features = obj.get("features", None)
    else:
        rf = obj; features = None
    if (features is None) and feats_path and os.path.exists(feats_path):
        with open(feats_path, "r") as f:
            meta = json.load(f)
            features = meta.get("features", None)
    if features is None:
        maybe = os.path.join(os.path.dirname(model_path), "features.json")
        if os.path.exists(maybe):
            with open(maybe, "r") as f:
                features = json.load(f).get("features", None)
    if features is None:
        raise RuntimeError("Could not recover feature list; provide --features path.")
    return rf, features

def metrics_from_pair(pred_clip, eta_clip):
    with rasterio.open(pred_clip) as dp, rasterio.open(eta_clip) as dn:
        p = dp.read(1).astype(np.float32); t = dn.read(1).astype(np.float32)
        ndp, ndt = dp.nodata, dn.nodata
    m = np.isfinite(p) & np.isfinite(t)
    if ndp is not None: m &= (p != ndp)
    if ndt is not None: m &= (t != ndt)
    if m.sum() == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, r=np.nan, r2=np.nan)
    e = p[m] - t[m]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e*e)))
    bias = float(np.mean(e))
    r = float(np.corrcoef(p[m], t[m])[0,1])
    return dict(n=int(m.sum()), mae=mae, rmse=rmse, bias=bias, r=r, r2=r*r)

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def run_any_aoi(aoi: str, date: str, model_path: str, feats_path: str | None,
                out_root: str = "wapor_20m_any_aoi", sa_json: str | None = None,
                show_inline: bool = False):
    """
    ONE date. Downloads stack for AOI+date, applies model, clips vs ETa300m, makes viz PNG.
    Returns dict with paths & quick metrics.
    """
    rf, features = _load_model_and_features(model_path, feats_path)
    stack_fp, aoi_name = ensure_stack_download(date, aoi, out_root=out_root, sa_json=sa_json)

    out_dir   = os.path.join(out_root, aoi_name)
    pred_dir  = os.path.join(out_dir, "preds")
    viz_dir   = os.path.join(out_dir, "viz")
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    base = os.path.basename(stack_fp)
    tag  = os.path.splitext(base)[0]  # aoi_..._YYYY-mm-dd_stack
    pred_tif  = os.path.join(pred_dir, f"{tag}_pred.tif")
    pred_clip = pred_tif.replace(".tif", "_clip.tif")
    eta_clip  = os.path.join(pred_dir, f"{tag}_eta300_clip.tif")
    png_path  = os.path.join(viz_dir,  f"{tag}_pred_vs_eta300.png")

    print("[PREDICT] Using features:", features)
    predict_stack_blockwise(stack_fp, rf, features, pred_tif, date_iso=date,
                            chunk=CHUNK_PRED, nodata=NODATA_OUT)
    clip_to_eta300_mask_pair(pred_tif, stack_fp, pred_clip, eta_clip, nodata=NODATA_OUT)
    visualise_minmax_robust(pred_clip, eta_clip, png_path,
                            title=f"{aoi_name} — {date}: Pred vs WaPOR ETa")

    m = metrics_from_pair(pred_clip, eta_clip)
    print("[METRICS vs ETa300m]", m)

    if show_inline:
        try:
            from IPython.display import Image as IPyImage, display as ipy_display
            ipy_display(IPyImage(filename=png_path))
        except Exception:
            pass

    return {
        "stack": stack_fp, "pred": pred_tif,
        "pred_clip": pred_clip, "eta300_clip": eta_clip,
        "png": png_path, "metrics": m
    }

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    _load_dotenv_if_present()
    ap = argparse.ArgumentParser(description="Apply trained RF model to ANY AOI for ONE dekad date.")
    ap.add_argument("--aoi", required=True, help="EE asset | bbox:minlon,minlat,maxlon,maxlat | /path/to/aoi.geojson")
    ap.add_argument("--date", required=True, help="Dekad start date YYYY-MM-DD (1st, 11th, or 21st)")
    ap.add_argument("--model", required=True, help="Path to rf_model.joblib")
    ap.add_argument("--features", default=None, help="Path to features.json (optional if embedded)")
    ap.add_argument("--out_root", default="wapor_20m_any_aoi", help="Output root folder")
    ap.add_argument("--sa_json", default=None, help="Service account JSON (optional; env/ADC used if omitted)")
    args = ap.parse_args()

    res = run_any_aoi(
        aoi=args.aoi, date=args.date,
        model_path=args.model, feats_path=args.features,
        out_root=args.out_root, sa_json=args.sa_json, show_inline=False
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
