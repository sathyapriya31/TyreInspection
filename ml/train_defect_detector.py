"""Train the tyre defect detector (YOLO) and export it for the Android app.

Input:  a YOLO-format dataset described by a data YAML (see tyre_defects.yaml).
Output: <out>/tyre_defects.tflite and <out>/defect_labels.txt

    python train_defect_detector.py --data tyre_defects.yaml
    python train_defect_detector.py --data tyre_defects.yaml --model yolo11n.pt --epochs 150

Class names are written to defect_labels.txt in model index order. Names the app knows
(bulge, cut, crack, puncture, sidewall_damage, exposed_cord, worn_tread, foreign_object,
uneven_wear) get specific severity and advice; any other name is treated as a MAJOR defect.
"""
import argparse
import pathlib
import shutil

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset YAML")
    p.add_argument("--model", default="yolov8n.pt", help="starting weights, e.g. yolov8n.pt / yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--int8", action="store_true", help="int8 quantization (smaller/faster, slight accuracy loss)")
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(__file__).parent.parent / "app/src/main/assets")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=25, project="runs", name="tyre_defects", exist_ok=True,
        # Tyres look the same flipped either way; rotation helps with sidewall photos.
        fliplr=0.5, flipud=0.5, degrees=15,
    )

    best = YOLO(pathlib.Path(model.trainer.save_dir) / "weights" / "best.pt")
    metrics = best.val(data=args.data)
    print(f"\nmAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")

    exported = best.export(format="tflite", imgsz=args.imgsz, int8=args.int8, data=args.data)

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, args.out / "tyre_defects.tflite")
    names = [best.names[i] for i in sorted(best.names)]
    (args.out / "defect_labels.txt").write_text("\n".join(names) + "\n")
    print(f"\nWrote {args.out / 'tyre_defects.tflite'} (from {exported})")
    print(f"Wrote {args.out / 'defect_labels.txt'}: {names}")


if __name__ == "__main__":
    main()
