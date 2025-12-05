from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HandwritingCleaner:
    """
    Lightweight public HF model that cleans raw OCR output.
    Works offline after first download.
    """

    def __init__(self):
        self.model_name = "t5-small"   # PUBLIC + small + CPU-friendly
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def clean(self, text: str) -> str:
        if not text or len(text.strip()) == "":
            return ""

        prompt = f"Correct OCR text and fix handwriting mistakes. Return only cleaned text:\n{text}"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

        cleaned = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return cleaned.strip()
