# train_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from services.cnn_model import TransferCNN
import json, os

IMG_SIZE = (128,128)
BATCH = 16

train_dir = "cnn_images/train"
val_dir   = "cnn_images/val"

train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Save class names for predictor
os.makedirs("models", exist_ok=True)
with open("models/class_names.json","w") as f:
    json.dump(class_names, f)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build and train model
model = TransferCNN(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(class_names))
model.summary()

model.train(train_ds, val_ds, epochs=10, out_path="models/cnn_transfer.keras")
print("Training complete. Model saved to models/cnn_transfer.keras")
