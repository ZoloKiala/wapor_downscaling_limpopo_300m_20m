# WaPOR 20 m Downscaling — Local
Everything you need is in this folder.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Provide your Earth Engine service account key JSON
export EE_SERVICE_ACCOUNT_FILE=''   # change to the path of service account file (e.g. tethys-app-1-acc3960d3dd6.json) (PowerShell): $env:EE_SERVICE_ACCOUNT_FILE="C:\path\key.json"

# Run the full pipeline (export → train → predict → metrics)
python wapor_downscale_20m_local.py

# tune with defaults (K=10, steps=12)
python wapor_downscale_20m_local_tuned.py --tune

# if stacks already exist
python wapor_downscale_20m_local_tuned.py --tune --skip-export

# tweak K or steps if needed
python wapor_downscale_20m_local_tuned.py --tune --filter-top-k 12 --forward-max-steps 10

# Optional: make summary plots
python plot_metrics_20m.py
```
# Baseline model (on new aoi)
python predict_any_aoi.py \
  --aoi "bbox:16.2,-29.1,17.7,-28.4" \
  --date 2020-01-01 \
  --model wapor_20m_local/outputs_rf20_auto/rf_model.joblib \
  --features wapor_20m_local/outputs_rf20_auto/features.json \
  --out_root wapor_20m_any_aoi


# Tuned model (on new aoi)
python predict_any_aoi.py \
  --aoi "bbox:16.2,-29.1,17.7,-28.4" \
  --date 2020-01-01 \
  --model wapor_20m_local/outputs_rf20_fs_fwd/rf_model.joblib \
  --features wapor_20m_local/outputs_rf20_fs_fwd/features.json \
  --out_root wapor_20m_any_aoi


## Outputs
- Local GeoTIFF stacks: `wapor_20m_local/BAIXO/`, `wapor_20m_local/LAMEGO/`
- Model + results: `wapor_20m_local/outputs_rf20_auto/`
