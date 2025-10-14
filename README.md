# WaPOR 20 m Downscaling — Local
Everything you need is in this folder.

## Quick start
```bash
# 1) Create and activate a virtualenv
python -m venv .venv && source .venv/bin/activate
# on Windows PowerShell:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Provide your Earth Engine service account key JSON
# Linux/macOS:
export EE_SERVICE_ACCOUNT_FILE="/path/to/tethys-app-1-acc3960d3dd6.json"
# Windows PowerShell:
# $env:EE_SERVICE_ACCOUNT_FILE="C:\path\to\tethys-app-1-acc3960d3dd6.json"
```

### Run the baseline end-to-end (export → train → predict → metrics)
```bash
python wapor_downscale_20m_local.py
```

### Run the tuned pipeline (filter + forward selection + optional RF tuning)
```bash
# Tune with defaults (K=10, max steps=12)
python wapor_downscale_20m_local_tuned.py --tune

# If stacks already exist locally
python wapor_downscale_20m_local_tuned.py --tune --skip-export

# Tweak shortlist size or max forward steps
python wapor_downscale_20m_local_tuned.py --tune --filter-top-k 12 --forward-max-steps 10
```

### Optional: make summary plots
```bash
# Baseline plots
python plot_metrics_20m.py

# Tuned plots (reads metrics_tuned.csv)
python plot_metrics_20m_tuned.py
```

---

## Apply the model to a new AOI (single dekad)

### Baseline model on a new AOI
```bash
python predict_any_aoi.py   --aoi "bbox:16.2,-29.1,17.7,-28.4"   --date 2020-01-01   --model wapor_20m_local/outputs_rf20_auto/rf_model.joblib   --features wapor_20m_local/outputs_rf20_auto/features.json   --out_root wapor_20m_any_aoi
```

### Tuned model on a new AOI
> Note: this script **does not print accuracy** (there is no native 20 m truth).  
> It can optionally **clip to the WaPOR 300 m footprint and produce a visualization PNG**.
```bash
python predict_any_aoi_tuned.py   --aoi "bbox:16.2,-29.1,17.7,-28.4"   --date 2020-01-01   --model wapor_20m_local/outputs_rf20_tuned/rf_model_tuned.joblib   --features wapor_20m_local/outputs_rf20_tuned/features_tuned.json   --out_root wapor_20m_any_aoi   --features-aware   --overwrite
```
- `--features-aware` ensures the downloaded stack contains **all predictors** your tuned model expects (fails fast if missing).
- `--overwrite` forces re-download of the stack if a file already exists.

---

## Outputs
- Local GeoTIFF stacks:  
  `wapor_20m_local/BAIXO/`, `wapor_20m_local/LAMEGO/`
- Baseline model + results:  
  `wapor_20m_local/outputs_rf20_auto/`
  - `rf_model.joblib`, `features.json`, `metrics.csv`, `preds/`, `viz/`
- **Tuned** model + results:  
  `wapor_20m_local/outputs_rf20_tuned/`
  - `rf_model_tuned.joblib`, `features_tuned.json`, `metrics_tuned.csv`, `preds/`, `viz/`
- Any-AOI predictions & PNGs:  
  `wapor_20m_any_aoi/<AOI_NAME>/preds/`, `wapor_20m_any_aoi/<AOI_NAME>/viz/`

---

## Notes & tips
- Auth alternatives (if you prefer not to set a file path):
  - `EE_SERVICE_ACCOUNT_JSON` (inline JSON), or
  - `EE_SERVICE_ACCOUNT_JSON_B64` (base64 of the JSON), or
  - prior `earthengine authenticate` (cached OAuth).
- A geedim warning like  
  `Couldn't find STAC entry for: 'None'`  
  can appear for some EE assets; it’s harmless when the file still saves.
- If you rerun frequently, add `--skip-export` on training scripts and `--overwrite` on the any-AOI script as needed.
