import pytesseract
from PIL import Image
import io

class OCRService:
    def extract_text(self, image_file):
        try:
            img = Image.open(image_file)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            print("Tesseract OCR failed:", e)
            return ""
