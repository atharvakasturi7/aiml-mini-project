# Road Damage Detection Using YOLOv8

## Overview
This academic mini‑project demonstrates an end‑to‑end computer‑vision pipeline for detecting **road surface damage** (longitudinal cracks, transverse cracks, alligator cracks, and potholes) using the **YOLOv8 Nano** model.

## Dataset
- **Road Damage Dataset 2022 (RDD2022)** – https://github.com/charliezjy/Road-Damage-Detection-2022
- License: CC‑BY‑4.0 (cite the original authors).
- **Download manually** and extract its contents so that:
  - Images → `data/raw_dataset/images/`
  - Annotations (XML or JSON) → `data/raw_dataset/annotations/`

## Target Classes (fixed mapping)
| ID | Label | RDD2022 Code |
|----|-------|--------------|
| 0  | Longitudinal Crack | D00 |
| 1  | Transverse Crack   | D10 |
| 2  | Alligator Crack    | D20 |
| 3  | Pothole            | D40 |

## Project Structure
```
road-damage-detection-yolov8/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ data.yaml               # YOLO dataset config
├─ check_installation.py
├─ train_yolo.py
├─ evaluate.py
├─ predict.py
├─ raw_dataset/            # <-- place original RDD2022 data here (not tracked)
│   ├─ images/
│   └─ annotations/
├─ dataset/                # YOLO‑ready split (generated)
│   ├─ images/
│   │   ├─ train/
│   │   └─ val/
│   └─ labels/
│       ├─ train/
│       └─ val/
├─ scripts/
│   ├─ convert_annotations.py   # converts XML/JSON → YOLO format & splits
│   └─ check_dataset.py          # validates image‑label consistency
├─ test_images/            # put a few unseen images for quick testing
├─ sample_results/
│   └─ predictions/        # inference outputs (annotated images & txt files)
└─ runs/                    # training & evaluation artefacts
```

## Installation (PowerShell)
```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1   # PowerShell
#   (or `source .venv/Scripts/activate` on Bash)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the environment
python check_installation.py
```
The code runs on CPU; a GPU will be used automatically if available.

## Dataset Preparation
```powershell
python scripts/convert_annotations.py
```
The script:
- Detects whether the source annotation is Pascal VOC XML or a simple JSON format.
- Converts bounding boxes to YOLO’s normalized `(class x_center y_center width height)` representation.
- Keeps **only** the four target classes (others are ignored and reported).
- Splits the data 80 %/20 % (train/val) with a fixed random seed.
- Places processed images and label files under `data/dataset/`.

You can inspect the conversion summary or run `scripts/check_dataset.py` for a sanity‑check.

## Training
```powershell
python train_yolo.py --epochs 5   # quick CPU smoke‑test
# optional arguments: --epochs N --imgsz SIZE --batch B --device cpu
```
Training artefacts (weights, logs, plots) are saved under `runs/road_damage_yolov8/`.

## Evaluation
```powershell
python evaluate.py
```
The script loads the best checkpoint (`runs/road_damage_yolov8/weights/best.pt`) and prints:
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Per‑class metrics
- Confusion matrix (saved in `runs/road_damage_yolov8/eval/`)

## Prediction & Visualization
```powershell
# Single image
python predict.py --source test_images/example.jpg

# Folder of images
python predict.py --source test_images
```
- Loads the trained model.
- Detects the four damage categories.
- Saves annotated images and YOLO‑style `.txt` label files under `sample_results/predictions/`.
- Adjust confidence threshold with `--conf-thres`.

## Limitations & Future Work
- CPU training is slow; use a GPU for full‑scale experiments.
- The converter currently supports XML and a simple JSON format; adapt it if the dataset layout changes.
- Only four damage types are modeled; additional classes can be added by extending `LABEL_MAP`.
- Post‑processing (NMS tuning, object tracking) is left to the user.

---
*Happy coding!*
