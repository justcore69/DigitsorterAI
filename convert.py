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
            threshold = 200
            if i == 1:
                print(img_array)
            
            height, width = img_array.shape

            # Iterate over all the pixels
            for x in range(width):
                for y in range(height):

                    if img_array[y, x] > threshold:
                        img_array[y, x] = 255
                    else:
                        img_array[y, x] = 0

            # Save the binary image
            Image.fromarray(img_array.astype(np.uint8)).save(f"train/train_conv_{i}.png")

            print(f"Converted {i} with {len(img.getbands())} bands; type: {img.mode}")