# services/hf_ocr.py

from paddleocr import PaddleOCR

class HFOCR:
    def __init__(self):
        # Lightweight CPU model (no GPU, safe for Codespaces)
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False
        )

    def extract_text(self, image_file):
        """
        Extracts readable text (printed or handwritten) from an uploaded file.
        """
        image_file.seek(0)
        img_bytes = image_file.read()

        # Save temporary image
        temp_path = "/tmp/tmp_ocr_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Run OCR
        result = self.ocr.ocr(temp_path)

        if not result or result[0] is None:
            return ""

        # Concatenate all detected text segments
        lines = [item[1][0] for item in result[0]]
        return "\n".join(lines).strip()
