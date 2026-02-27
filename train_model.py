"""
AI vs Real Image Detector — Training Script
Compatible with TensorFlow 2.18+ / Keras 3.x
Uses EfficientNetB4 with transfer learning on the CIFAKE dataset.

Usage:
    Full training  :  python train_model.py
    Quick test run :  python train_model.py --quick
"""

import os
import sys
import json
import argparse
import numpy as np

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras import layers, Model, callbacks
from keras.applications import EfficientNetB4
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── ARGS ─────────────────────────────
parser = argparse.ArgumentParser(description="Train AI vs Real Image Detector")
parser.add_argument("--quick", action="store_true",
                    help="Quick test run with reduced data and smaller image size")
args = parser.parse_args()

QUICK = args.quick

# ─────────────────────────── CONFIG ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR   = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnetb4_ai_detector.keras")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

if QUICK:
    IMG_SIZE   = 160        # Smaller for fast testing
    BATCH_SIZE = 16
    EPOCHS     = 2
    LR         = 1e-3
    print("⚡ QUICK MODE — reduced resolution, epochs, and data")
else:
    IMG_SIZE   = 380        # EfficientNetB4 default
    BATCH_SIZE = 16
    EPOCHS     = 10
    LR         = 1e-4

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────── DATA LOADING ─────────────────────
print("[INFO] Loading training data …")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.15,
    subset="training",
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    validation_split=0.15,
    subset="validation",
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Class names (alphabetical: FAKE=0, REAL=1)
print(f"[INFO] Class names: {train_ds.class_names}")

# --- If QUICK mode, take a small slice of data ---
if QUICK:
    train_ds = train_ds.take(50)   # ~800 images
    val_ds   = val_ds.take(15)     # ~240 images
    test_ds  = test_ds.take(20)    # ~320 images
    print("[INFO] QUICK mode: using ~800 train, ~240 val, ~320 test images")

# Rescale pixels to [0, 1] and prefetch
rescale = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

# Data augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# ─────────────────────────── BUILD MODEL ──────────────────────
print("[INFO] Building EfficientNetB4 model …")

base_model = EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze the base initially
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu", name="fc1")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

model = Model(inputs, outputs, name="AI_vs_Real_Detector")
model.summary()

# ─────────────────────────── PHASE 1: Train head ─────────────
print("\n[PHASE 1] Training classifier head (base frozen) …")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
)

# ─────────────────────────── PHASE 2: Fine-tune top layers ───
print("\n[PHASE 2] Fine-tuning top layers of EfficientNetB4 …")
base_model.trainable = True

# Freeze everything except the last 30 layers
for layer_item in base_model.layers[:-30]:
    layer_item.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR / 10),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
)

# ─────────────────────────── EVALUATE ─────────────────────────
print("\n[INFO] Evaluating on test set …")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# Gather predictions and true labels
all_preds = []
all_labels = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    all_preds.append(preds)
    all_labels.append(labels.numpy())

all_preds  = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0).flatten()
pred_classes = (all_preds > 0.5).astype(int).flatten()

print("\nClassification Report:")
try:
    print(classification_report(
        all_labels.astype(int), pred_classes,
        target_names=["FAKE", "REAL"],
        labels=[0, 1],
        zero_division=0,
    ))
except Exception as e:
    print(f"(Could not generate full report: {e})")

print("Confusion Matrix:")
print(confusion_matrix(all_labels.astype(int), pred_classes, labels=[0, 1]))

# ─────────────────────────── SAVE MODEL ───────────────────────
model.save(MODEL_PATH)
print(f"\n[INFO] Model saved to {MODEL_PATH}")

# Save metadata (image size used) for the Streamlit app
meta = {"img_size": IMG_SIZE, "class_names": ["FAKE", "REAL"]}
with open(os.path.join(BASE_DIR, "model", "model_meta.json"), "w") as f:
    json.dump(meta, f)
print("[INFO] Model metadata saved")

# ─────────────────────────── PLOTS ────────────────────────────
def plot_history(h1, h2, metric, title, filename):
    """Combine Phase-1 and Phase-2 history and plot."""
    vals = h1.history[metric] + h2.history[metric]
    val_vals = h1.history[f"val_{metric}"] + h2.history[f"val_{metric}"]
    epochs_range = range(1, len(vals) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, vals, label=f"Train {metric}")
    plt.plot(epochs_range, val_vals, label=f"Val {metric}")
    plt.axvline(len(h1.history[metric]) + 0.5, color="gray", linestyle="--", label="Fine-tune start")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()
    print(f"[INFO] Saved plot: {filename}")

plot_history(history_head, history_fine, "accuracy", "Model Accuracy", "accuracy.png")
plot_history(history_head, history_fine, "loss", "Model Loss", "loss.png")

print("\n✅ Training complete!")
