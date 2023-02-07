from PIL import Image 
from PIL import ImageOps
import numpy as np

def convert_images():
    for i in range(45):
        if i != 0:
            img = Image.open(f"train/train_conv_{i}.png") 
            img = img.convert('L') 
            img = img.resize((28, 28))

            img_array = np.array(img)

            # Set the threshold value
            threshold = 4
            if i == 1:
                print(img_array)
            # Create the binary image
            binary_img = (img_array > threshold) * 1

            # Save the binary image
            Image.fromarray(binary_img.astype(np.uint8)).save(f"train/train_conv_{i}.png")

            #print(f"Converted {i} with {len(img.getbands())} bands; type: {img.mode}")