# Model weights (Keras `.keras`)

Saved TensorFlow Keras models from `notebooks/02_baseline_cnn.ipynb`:

| File | Role |
|------|------|
| `baseline_cnn_mrl.keras` | Baseline CNN |
| `cnn_mrl_driver_aug.keras` | Augmented training (default for inference in `src/drowsiness/config.py`) |
| `cnn_mrl_driver_aug_deep.keras` | Deeper augmented CNN |

Copy the file you need into `models/` (or update `DEFAULT_MODEL_PATH` in `src/drowsiness/config.py`).

**GitHub Pages** (model evaluation + download links): enable **Settings → Pages → Branch `main` / folder `docs`**, then open the site linked from the main README.
