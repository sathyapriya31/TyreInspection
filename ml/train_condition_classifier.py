"""Train the tyre condition classifier and export it for the Android app.

Input:  a folder with one sub-folder per class, e.g.
            data/condition/defective/*.jpg
            data/condition/good/*.jpg
        (the Kaggle "Tyre Quality Classification" dataset has exactly this layout).
Output: <out>/tyre_condition.tflite and <out>/condition_labels.txt

The model is MobileNetV3-Small with Keras' built-in preprocessing, so it takes raw
0..255 RGB float input. That is what TyreInspector.classify() feeds it.

    python train_condition_classifier.py --data data/condition
"""
import argparse
import pathlib
import tempfile

import numpy as np
import tensorflow as tf

IMG = 224


def build_datasets(data_dir: pathlib.Path, batch: int):
    common = dict(
        validation_split=0.2, seed=42, image_size=(IMG, IMG),
        batch_size=batch, label_mode="categorical",
    )
    train = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)
    names = train.class_names
    autotune = tf.data.AUTOTUNE
    return train.cache().shuffle(1000).prefetch(autotune), val.cache().prefetch(autotune), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(__file__).parent.parent / "app/src/main/assets")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--quantize", action="store_true",
                   help="dynamic-range quantization: ~4x smaller, float input/output kept")
    args = p.parse_args()

    train, val, names = build_datasets(args.data, args.batch)
    print("Classes:", names)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)),
    ], name="augment")
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet", include_preprocessing=True,
    )
    pool = tf.keras.layers.GlobalAveragePooling2D()
    dropout = tf.keras.layers.Dropout(0.3)
    head = tf.keras.layers.Dense(len(names), activation="softmax")

    inputs = tf.keras.Input((IMG, IMG, 3))
    outputs = head(dropout(pool(base(augment(inputs), training=False))))
    model = tf.keras.Model(inputs, outputs)

    stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

    # Stage 1: train only the new head.
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[stop])

    # Stage 2: fine-tune the top of the backbone with a small learning rate.
    base.trainable = True
    for layer in base.layers[: int(len(base.layers) * 0.6)]:
        layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-5), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.fine_tune_epochs, callbacks=[stop])

    # Validation report.
    y_true, y_pred = [], []
    for x, y in val:
        y_true += list(np.argmax(y, axis=1))
        y_pred += list(np.argmax(model.predict(x, verbose=0), axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(f"\nValidation accuracy: {np.mean(y_true == y_pred):.3f}")
    print("Confusion matrix (rows = true, cols = predicted):", names)
    print(tf.math.confusion_matrix(y_true, y_pred, num_classes=len(names)).numpy())

    # Inference graph without augmentation/dropout.
    infer_in = tf.keras.Input((IMG, IMG, 3), name="image")
    infer = tf.keras.Model(infer_in, head(pool(base(infer_in, training=False))))

    with tempfile.TemporaryDirectory() as tmp:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(infer)
            if args.quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite = converter.convert()
        except Exception:  # Keras 3: go through a SavedModel export instead
            infer.export(tmp)
            converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
            if args.quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite = converter.convert()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tyre_condition.tflite").write_bytes(tflite)
    (args.out / "condition_labels.txt").write_text("\n".join(names) + "\n")
    print(f"\nWrote {args.out / 'tyre_condition.tflite'} ({len(tflite) / 1e6:.1f} MB)")
    print(f"Wrote {args.out / 'condition_labels.txt'}: {names}")


if __name__ == "__main__":
    main()
