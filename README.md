# Drowsiness Detection from Eye Images (Deep Learning)

**Per-frame binary classifier** that predicts **awake vs. sleepy** from **grayscale infrared eye crops**, trained on the **MRL Eye Dataset**, with optional **offline video** and **webcam** demos that combine **dlib** face landmarks with the trained **Keras** model.

---

## The 2-minute recruiter brief

| Question | Answer |
|----------|--------|
| **What did you build?** | An end-to-end pipeline: EDA and `tf.data` loading → CNN training → evaluation (accuracy, report, confusion matrix, ROC) → inference scripts for recorded video and live camera with temporal “sleepy” logic. |
| **What data did you use?** | Primarily the **MRL Eye Dataset** (infrared, ~85k images in EDA notebook; test eval on **16,981** held-out images, 2 classes). A second Kaggle **Eye dataset** (RGB open/closed) is documented for extension. |
| **How did you build it?** | **TensorFlow/Keras** CNN on **64×64×1** inputs, train/val/test folders, augmentation during training, **scikit-learn** for metrics, **OpenCV** + **dlib** for face/eye crops at inference. |
| **What was the result?** | **~98.6% test accuracy** on the baseline CNN run (see **Evaluation** below). Training curves, confusion matrix, and ROC are produced in the modeling notebook. |

---

## Problem statement

Driver **drowsiness** is a major safety risk. This project focuses on a **proxy task**: from a **crop of the eye region**, classify whether the subject appears **awake** or **sleepy** (eyes open vs. closed / drowsy pattern). That supports downstream applications such as in-cabin alerts, provided real-world validation is done separately.

---

## Tech stack

| Layer | Tools |
|--------|--------|
| **Deep learning** | TensorFlow / Keras (`tensorflow>=2.13`) |
| **Data & EDA** | pandas, NumPy, Matplotlib |
| **Classical ML metrics** | scikit-learn (report, confusion matrix, ROC/AUC) |
| **Vision / I/O** | OpenCV, Pillow |
| **Inference (face/eyes)** | dlib (face detection + 68-point landmarks), OpenCV |
| **Environment** | Python 3.x, `requirements.txt` (see **Setup**) |

---

## Dataset & sources

Data are **not** committed to the repo (large files stay local). Download and arrange as follows.

### 1. MRL Eye Dataset (infrared) — primary training source

- **Kaggle:** [MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)
- **Layout:**

```text
data/
  MRL/
    data/
      train/    # subfolders per class, e.g. awake / sleepy
      val/
      test/
```

### 2. Eye Dataset (open / closed eyes) — optional / secondary

- **Kaggle:** [Eye Dataset](https://www.kaggle.com/datasets/prasadvpatil/eye-dataset)
- **Layout:**

```text
data/
  Eye dataset/
    train_dataset/
      openLeftEyes/
      openRightEyes/
      closedLeftEyes/
      closedRightEyes/
```

---

## Project structure

```text
Deep-Learning-Drowsiness-Prediction_1/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_gathering.ipynb   # EDA, class balance, tf.data pipeline
│   └── 02_baseline_cnn.ipynb     # CNN train/eval, metrics, save weights
├── src/
│   └── drowsiness/
│       ├── config.py             # Paths, thresholds, IMG_SIZE
│       ├── preprocess.py         # Eye crop → model tensor
│       └── eye_crop.py           # dlib face / eye regions
├── scripts/
│   ├── detect_video.py         # Offline video → annotated MP4 + crops
│   └── detect_live.py          # Webcam demo
├── data/                       # Datasets (gitignored); see Dataset section
├── models/                     # *.keras weights (gitignored except .gitkeep)
├── assets/                     # shape_predictor_68_face_landmarks.dat (gitignored)
└── outputs/                    # videos/ + crops/ from scripts (gitignored)
```

---

## EDA summary

*(From `notebooks/01_data_gathering.ipynb`.)*

- **MRL infrared data:** on the order of **~85k** images total in the EDA run; paths collected under `data/MRL/data`.
- **Splits:** counts and **class ratios** per **train / val / test** for labels such as **awake** vs. **sleepy** (see notebook tables).
- **Pipeline:** `tf.keras.utils.image_dataset_from_directory`, grayscale, resize, normalization, light **data augmentation** (for modeling), `tf.data` prefetch/shuffle patterns.

---

## Modeling approach

*(From `notebooks/02_baseline_cnn.ipynb`.)*

- **Task:** Binary image classification on **64×64 grayscale** eye patches (`input_shape` → `(64, 64, 1)`).
- **Model:** Convolutional network (`baseline_cnn_mrl`): stacked **Conv2D / pooling** blocks, dense head, **sigmoid** for binary output.
- **Training:** `model.fit` with **Adam**, **binary cross-entropy**, monitored **val** performance; **early stopping** available in notebook flow; training capped at **15 epochs** in the logged run (best val metrics in mid-to-late epochs).
- **Artifacts:** Notebook saves under `models/` (e.g. `baseline_cnn_mrl.keras`, `cnn_mrl_driver_aug.keras`).

---

## Evaluation metrics & results

Reported on the **held-out test** split (**16,981** images, **2** classes) in the baseline notebook:

| Metric | Approximate result (baseline run) |
|--------|-----------------------------------|
| **Test accuracy** | **~0.986** (98.6%) |
| **Precision / recall / F1** | **~0.99** macro / weighted (see notebook `classification_report`) |
| **Confusion matrix** | Printed in notebook |
| **ROC / AUC** | ROC curve plotted; AUC in figure label (see notebook) |

The notebook also discusses **hard examples** and **threshold** behavior (probability cutoffs for “sleepy”) for operational use.

### Results / output screenshots

- **Training curves** (loss & accuracy vs. epoch) and **ROC** plots are **inside** `notebooks/02_baseline_cnn.ipynb` as cell outputs.
- For a portfolio README, export those figures and add them here, for example:

```text
docs/images/training_curves.png
docs/images/confusion_matrix.png
docs/images/roc_curve.png
```

*(Create a `docs/images/` folder, save exports from the notebook, and link them in this section with standard Markdown image syntax.)*

---

## Sample output

- **Training:** Keras progress logs show **val_accuracy** reaching roughly **~0.986** in later epochs (full logs in `notebooks/02_baseline_cnn.ipynb`).
- **Offline video (`scripts/detect_video.py`):** loads `models/cnn_mrl_driver_aug.keras` (see `src/drowsiness/config.py`), reads `data/test_input/test_video.mp4`, writes **`outputs/videos/<timestamp>.mp4`** and eye crops under **`outputs/crops/<timestamp>/`**.
- **Live demo (`scripts/detect_live.py`):** webcam overlay with **awake / sleepy** after sustained eye-closure (default **5 s** above probability threshold).

**Note:** The baseline notebook may save **`models/baseline_cnn_mrl.keras`** first; inference defaults to **`models/cnn_mrl_driver_aug.keras`**. After training, **copy or rename** the weights, or set **`DEFAULT_MODEL_PATH`** in `src/drowsiness/config.py`.

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes: TensorFlow, NumPy, pandas, Matplotlib, scikit-learn, OpenCV, Pillow, tqdm, **dlib** (may need build tools or a prebuilt wheel on your OS).

Download **dlib’s** `shape_predictor_68_face_landmarks.dat` into **`assets/`** (see `src/drowsiness/config.py`).

### 3. Data

Place **MRL** (and optional **Eye dataset**) under **`data/`** as described above.

---

## How to run

From the **project root** (this folder), with `src` on the path for scripts (handled inside `scripts/`).

1. **EDA:** Run `notebooks/01_data_gathering.ipynb` (kernel cwd can be project root or `notebooks/`; the notebook adjusts).
2. **Train & evaluate:** Run `notebooks/02_baseline_cnn.ipynb`; weights go under `models/`.
3. **Video inference:** Put a sample clip at `data/test_input/test_video.mp4`, ensure `models/cnn_mrl_driver_aug.keras` exists, then:

   `python scripts/detect_video.py`

4. **Live webcam:** `python scripts/detect_live.py` (camera permissions).

If you use a **disposable VM**, recreate the venv and reinstall dependencies before re-running.

---

## Conclusion & recommendations

- The **baseline CNN** achieves **strong test-set metrics** on **curated MRL splits**, which is a solid proof of concept for **eye-state** classification.
- **Deployment caveats:** Real vehicles need checks on **lighting, pose, glasses, IR vs. RGB**, and **temporal smoothing** different from a static test set; consider **calibration** on your own video and **error analysis** on false positives (alert fatigue).
- **Next steps:** Try **temporal models** (e.g. CNN + RNN/Transformer on frame sequences), **multi-dataset** training with the second Kaggle set, and **on-device** latency tests.

---

## Takeaway

This repository shows a **complete ML story**: public data → **EDA** → **CNN** training → **rigorous evaluation** (accuracy, report, confusion matrix, ROC) → **practical inference** (video + webcam) with explicit **engineering notes** (paths, dlib binary, model filename alignment). Keep **figures** in the README updated from the notebook exports so reviewers never need to execute code to see the headline results.
