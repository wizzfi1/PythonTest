import easyocr
import numpy as np
from PIL import Image
import io

class EasyOCRService:
    def __init__(self):
        # English only, CPU only, no GPU issues
        self.reader = easyocr.Reader(["en"], gpu=False)

    def extract_text(self, image_file):
        """Extract text using EasyOCR from PIL or file upload."""

        image_bytes = image_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(pil_img)

        results = self.reader.readtext(img_arr, detail=0)

        if not results:
            return ""

        # Join all OCR segments into one line
        extracted = " ".join(results).strip()

        return extracted
