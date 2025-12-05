import base64
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AIOCR:
    def extract_text(self, image_file):
        # Read image bytes
        img_bytes = image_file.read()
        b64 = base64.b64encode(img_bytes).decode()

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ONLY the handwritten text. No explanations."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{b64}"
                        }
                    ]
                }
            ]
        )

        # Correct extraction
        return response.choices[0].message["content"]
