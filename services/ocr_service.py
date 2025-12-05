import easyocr
import numpy as np
from PIL import Image

class EasyOCRService:
    def __init__(self):
        # English + numerical handwriting support
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_file):
        """Extract handwritten text using EasyOCR."""
        try:
            image = Image.open(image_file).convert("RGB")
            img_np = np.array(image)

            results = self.reader.readtext(img_np)

            # Join detected text
            extracted = " ".join([r[1] for r in results]).strip()
            return extracted
        except Exception as e:
            print("EasyOCR error:", e)
            return ""
