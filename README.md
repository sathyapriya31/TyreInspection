# Tyre Inspection (Android)

An Android app that inspects vehicle tyres with the phone camera and on-device AI models (LiteRT / TensorFlow Lite). It needs no internet connection and no server.

## Features

- **Live camera inspection:** defects are boxed on the preview in real time, with a running verdict.
- **Detailed report:** take a photo or pick one from the gallery to get:
  - a verdict: PASS, ATTENTION, FAIL, RETAKE or UNKNOWN
  - the overall condition (classifier class and confidence)
  - each defect with its severity: critical, major or minor
  - repair and replace advice for each defect type
  - a photo-quality check (too dark, overexposed or blurry triggers RETAKE)
- **Tyre position** tagging: FL, FR, RL, RR or spare.
- **History** of inspections stored on the device, with **share** (photo and text report).
- Torch toggle for wheel arches and poor light.

## Project layout

```
app/src/main/java/com/spritle/tyreinspection/
  ml/
    TyreInspector.kt   runs both models on a photo and builds the InspectionResult
    TfliteModel.kt     LiteRT wrapper (float/uint8/int8, NHWC/NCHW)
    YoloDecoder.kt     YOLOv8/YOLO11 output decoding + class-aware NMS
    VerdictRules.kt    severity table, verdict logic, advice text
    ImageUtils.kt      rotation, gallery decoding, brightness/blur metrics
    Models.kt          result data types
  data/InspectionRepository.kt   JPEG + JSON storage of inspections
  ui/                            MainActivity (camera), ResultActivity, HistoryActivity, OverlayView
app/src/main/assets/             models + label files go here
ml/                              training and export scripts (see ml/README.md)
```

## AI models

The app looks for two optional models in `app/src/main/assets/`:

| File | Purpose |
|---|---|
| `tyre_condition.tflite` + `condition_labels.txt` | whole-photo condition (good / defective) |
| `tyre_defects.tflite` + `defect_labels.txt` | YOLO detector that localises defects |

**No trained models are included.** Train them with the scripts in [`ml/`](ml/README.md); each script writes its model into the assets folder. Without models the app still runs, shows a banner, and reports UNKNOWN.

### How the verdict is decided (`VerdictRules.kt`)

1. If no model is installed, the verdict is **UNKNOWN**.
2. If the photo is dark, overexposed or blurry, the verdict is **RETAKE**, unless a critical defect is clearly visible.
3. A critical defect (bulge, cut, sidewall damage, puncture or exposed cord), or the classifier rating the tyre defective at 85% or more, gives **FAIL**.
4. A major defect (crack, worn tread or foreign object), or the classifier rating it defective at 50% or more, gives **ATTENTION**.
5. Otherwise the verdict is **PASS**.

Detector labels not in the table are treated as major, which errs on the side of safety. You can tune the thresholds at the top of `VerdictRules.kt` and in `TyreInspector.DETECTION_THRESHOLD`.

## Build and run

Requirements: Android Studio (or the Android SDK with platform 34) and JDK 17. Devices need Android 8.0 (API 26) or later.

```bash
./gradlew assembleDebug                 # APK -> app/build/outputs/apk/debug/app-debug.apk
./gradlew installDebug                  # install on a connected device
./gradlew testDebugUnitTest             # unit tests for verdict rules + YOLO decoding
```

Or open the folder in Android Studio and press Run.

## Limitations

- A camera cannot measure **tread depth** in millimetres reliably. The app flags visibly worn tread, and the advice tells the user to confirm with a depth gauge.
- Accuracy depends entirely on the training data. Validate the models on your own tyres and lighting conditions before relying on them.
- Results are guidance only, not a substitute for inspection by a qualified technician.
