
import os
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib

# Use a non-interactive backend to save graphs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks

# -----------------------------------
# 1. SETTINGS
# -----------------------------------

DATA_DIR = Path("Dataset/Training")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

MODEL_PATH = "sign_language_model.keras"
LABELS_PATH = "labels.txt"

# -----------------------------------
# 2. CHECK DATASET
# -----------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}\n"
        "Please check the DATA_DIR path."
    )

class_folders = [
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir()
]

if len(class_folders) < 2:
    raise ValueError(
        "The dataset must contain at least two class folders."
    )

print("Dataset path:", DATA_DIR.resolve())
print("Number of classes:", len(class_folders))

# -----------------------------------
# 3. LOAD DATASET
# -----------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
print(class_names)

# -----------------------------------
# 4. DISPLAY SAMPLE IMAGES
# -----------------------------------

for images, labels in train_ds.take(1):

    plt.figure(figsize=(10, 10))

    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        label_index = int(labels[i].numpy())
        plt.title(class_names[label_index])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150)
    plt.close()

print("Sample images saved to sample_images.png")

# -----------------------------------
# 5. NORMALIZATION AND PERFORMANCE
# -----------------------------------

normalization_layer = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------------
# 6. BUILD CNN MODEL
# -----------------------------------

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Convolutional block 1
    layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional block 2
    layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional block 3
    layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Feature extraction
    layers.Flatten(),

    # Classification layers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])

# -----------------------------------
# 7. DISPLAY MODEL
# -----------------------------------

model.summary()

# -----------------------------------
# 8. COMPILE MODEL
# -----------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------
# 9. TRAINING CALLBACKS
# -----------------------------------

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# -----------------------------------
# 10. TRAIN MODEL
# -----------------------------------

print("\nTraining started...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# -----------------------------------
# 11. EVALUATE MODEL
# -----------------------------------

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\nValidation Loss:", round(loss, 4))
print("Validation Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------------
# 12. PLOT ACCURACY
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("training_accuracy.png", dpi=150)
plt.close()

print("Accuracy graph saved to training_accuracy.png")

# -----------------------------------
# 13. PLOT LOSS
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("training_loss.png", dpi=150)
plt.close()

print("Loss graph saved to training_loss.png")

# -----------------------------------
# 14. SAVE MODEL
# -----------------------------------

model.save(MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")

# -----------------------------------
# 15. SAVE CLASS NAMES
# -----------------------------------

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"Labels saved successfully: {LABELS_PATH}")

# -----------------------------------
# 16. COMPLETION
# -----------------------------------

print("\nProject completed successfully!")
