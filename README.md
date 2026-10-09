# HydroTwin daily model retraining

Every day, this repo teaches the HydroTwin river models the newest data and
publishes them so the WebXR river pages always forecast with fresh models.

## How it works (plain words)

1. **Fetch the new days** (`refresh_data.py`) — downloads the newest daily
   river flow (USGS) and weather (Open-Meteo) for all 14 rivers and appends
   them to `data/daily/`. History is never rewritten. (Weather runs ~5 days
   behind, so each run usually adds about one usable day.)
2. **Re-teach the models** (`train_export.py`) — retrains the 4 neural models
   (LSTM, Transformer, TFT, PatchTST) on the longer history and saves
   checkpoints to `ml/checkpoints/`. Resume-safe: finished stations are skipped.
3. **Convert for the browser** (`export_onnx.py`) — converts each trained model
   to ONNX format, which runs directly in a web page.
4. **Publish** — the ONNX files go to GitHub Pages at
   `https://saurabhkulkarni77.github.io/hydrotwin-models/onnx/<site>/<Model>.onnx`.
   The WebXR station pages load them from there, so they're always current
   with zero redeploys.
5. **Save the data** — the refreshed CSVs are committed back to the repo.

## Schedule

Runs automatically every day around 3am US Central (`.github/workflows/daily-retrain.yml`).
You can also trigger it manually from the Actions tab ("Run workflow").

## Files

- `refresh_data.py` — daily data refresh
- `train_export.py` — trains the 4 models, saves checkpoints (+ per-station stats)
- `export_onnx.py` — checkpoints → ONNX (`--out` sets the output dir)
- `best_lr.json` — validation-picked learning rates per model per station
- `data/daily/` — 14 river CSVs (grows daily)
- `train_lstm_normal.py`, `optimize_tr_gcn.py`, `train_sota8.py`, `train_new_arch.py` — model definitions + training protocol
