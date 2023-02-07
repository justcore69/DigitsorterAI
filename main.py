import PIL
from PIL import Image 
from PIL import ImageOps
import os
import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import convert

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)

CREATING_MODE = False
EPOCHS = 50

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=EPOCHS)
    model.save('digitsorter.nnm')

def load_model():
    model = tf.keras.models.load_model('digitsorter.nnm')
    loss, accuracy = model.evaluate(train_x, train_y)
    print(loss)
    print(accuracy)

    image_index = 1

    convert.convert_images()

    while os.path.isfile(f"train/train_conv_{image_index}.png"):
        img = cv2.imread(f"train/train_conv_{image_index}.png")
        img = np.invert(np.array([img]))

        image = np.expand_dims(np.mean(img.reshape(28, 28, 3).astype('float32') / 255.0, axis=2, keepdims=False), axis=0)

        pred = model.predict(image)
        print(f"This digit may be {np.argmax(pred)}")

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

        image_index+=1

def main():
    if CREATING_MODE:
        create_model()
    else:
        load_model()

if __name__ == "__main__":
    main()