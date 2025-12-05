import tensorflow as tf
import numpy as np
from PIL import Image
import json

class CNNPredictor:
    def __init__(self, model_path="models/cnn_transfer.keras",

                 class_json="models/class_names.json",
                 img_size=(128,128)):
        self.img_size = img_size
        self.model = tf.keras.models.load_model(model_path)
        with open(class_json, "r") as f:
            self.class_names = json.load(f)

    def preprocess(self, image_file):
        img = Image.open(image_file).convert("RGB")
        img = img.resize(self.img_size)
        arr = np.array(img).astype("float32")
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return arr

    def predict(self, file, top_k=1):
        arr = self.preprocess(file)
        preds = self.model.predict(np.expand_dims(arr, axis=0))[0]

        idxs = preds.argsort()[-top_k:][::-1]
        return {
            "class": self.class_names[idxs[0]],
            "score": float(preds[idxs[0]])
        }
