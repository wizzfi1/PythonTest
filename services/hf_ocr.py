import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

class HFOCR:
    def __init__(self):
        print("Loading Nougat OCR model…")
        self.processor = AutoProcessor.from_pretrained("facebook/nougat-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

    def extract_text(self, file):
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

            generated_ids = self.model.generate(
                pixel_values,
                max_length=256,
                num_beams=4
            )

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()

        except Exception as e:
            print("HF OCR failed:", e)
            return ""
