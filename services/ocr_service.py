import pytesseract
from PIL import Image
import io

class OCRService:
    def extract_text(self, image_file):
        """
        Extract text from an image uploaded via Flask.
        """
        # Read image into PIL format
        img = Image.open(image_file)

        # Use Tesseract to extract text
        extracted = pytesseract.image_to_string(img)

        # Clean text
        extracted = extracted.strip()

        return extracted
