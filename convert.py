from PIL import Image 
from PIL import ImageOps
import numpy as np

for i in range(45):
    if i != 0:
        img = Image.open(f"train/train_conv_{i}.png") 
        img = img.convert('L') 
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img.save(f"train/train_conv_{i}.png") 
        print(f"Converted {i} with {len(img.getbands())} bands; type: {img.mode}")