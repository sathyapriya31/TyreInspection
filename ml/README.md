# Training the tyre models

The app uses two optional on-device models. Each one works on its own, and the app is more accurate with both.

| Model | File in `app/src/main/assets/` | What it does | Script |
|---|---|---|---|
| Condition classifier | `tyre_condition.tflite` + `condition_labels.txt` | Whole-photo verdict, e.g. `good` / `defective` | `train_condition_classifier.py` |
| Defect detector | `tyre_defects.tflite` + `defect_labels.txt` | Boxes around bulges, cuts, cracks, punctures and other defects | `train_defect_detector.py` |

Both scripts write straight into the app's assets folder. Rebuild the app afterwards.

## Setup

```bash
cd ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU helps but is not required. Classifier training takes minutes on a CPU; YOLO training is much faster on a GPU (Google Colab works).

## 1. Condition classifier

**Data:** one folder per class:

```
data/condition/defective/*.jpg
data/condition/good/*.jpg
```

To get started quickly, use the Kaggle dataset **"Tyre Quality Classification"** (about 1,850 images labelled defective and good, already in this layout). You can add more classes, such as `worn`. Labels named `defective`, `damaged`, `bad`, `worn` or `unsafe` count as defective in the verdict rules.

```bash
python train_condition_classifier.py --data data/condition            # float32, ~4 MB
python train_condition_classifier.py --data data/condition --quantize # ~1 MB
```

The script prints validation accuracy and a confusion matrix. Aim for at least 90% on your own photos before trusting the model.

## 2. Defect detector (YOLO)

**Data:** YOLO-format bounding boxes. Search Roboflow Universe for "tire defect" or "tyre damage" for public datasets, or label your own photos with Roboflow, CVAT or Label Studio. Then edit `tyre_defects.yaml` so `names` matches your classes.

```bash
python train_defect_detector.py --data tyre_defects.yaml                       # YOLOv8n
python train_defect_detector.py --data tyre_defects.yaml --model yolo11n.pt    # YOLO11n
python train_defect_detector.py --data tyre_defects.yaml --int8                # quantized
```

## What the app expects

- **Classifier:** NHWC RGB input, any size (read from the model), raw **0–255** values. The app applies softmax if the output is logits.
- **Detector:** Ultralytics TFLite export, RGB input in **0–1**, output `[1, 4+classes, anchors]` or `[1, anchors, 4+classes]`, with normalized or pixel coordinates.
- Float32, UINT8 and INT8 tensors are all supported. The app quantizes inputs and dequantizes outputs using each tensor's own parameters.

If you change the preprocessing (for example, a classifier that expects −1..1), update `pixelDivisor` in `TyreInspector.kt` to match.

## Tips for good results

- Train on photos taken the way the app is used: phone camera, 30–50 cm from the tyre, both tread and sidewall views, in daylight and under torchlight.
- Include plenty of **good** tyres, especially dirty or wet ones, so mud isn't mistaken for damage.
- Keep a test set of tyres from vehicles that never appear in training.
