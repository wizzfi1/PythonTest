# scripts/generate_demo_images.py
import os
from PIL import Image, ImageDraw
import random

OUT = "cnn_images"
classes = ["red_mug","blue_mug","green_mug"]
sizes = (128,128)
per_class = 40  # images per class for demo

os.makedirs(OUT, exist_ok=True)
for split in ["train","val","test"]:
    for c in classes:
        os.makedirs(os.path.join(OUT, split, c), exist_ok=True)

def make_image(color, noise=False):
    img = Image.new("RGB", sizes, (255,255,255))
    draw = ImageDraw.Draw(img)
    # draw a colored circle / mug-like shape
    cx, cy = sizes[0]//2, sizes[1]//2
    r = random.randint(24,44)
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color)
    if noise:
        for _ in range(300):
            x = random.randint(0, sizes[0]-1)
            y = random.randint(0, sizes[1]-1)
            img.putpixel((x,y),(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    return img

# generate
for cls in classes:
    color = (255,50,50) if cls=="red_mug" else ((50,50,255) if cls=="blue_mug" else (50,200,50))
    for i in range(per_class):
        img = make_image(color, noise=(i%5==0))
        # split: first 32 -> train, next 4 -> val, last 4 -> test
        if i < 32:
            split="train"
        elif i < 36:
            split="val"
        else:
            split="test"
        path = os.path.join(OUT, split, cls, f"{cls}_{i}.jpg")
        img.save(path, quality=85)
print("Demo images generated at:", OUT)
