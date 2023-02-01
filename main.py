import PIL
from PIL import Image 
import os
import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = tf.keras.utils.normalize(train_x, axis=1)
test_x = tf.keras.utils.normalize(test_x, axis=1)

CREATING_MODE = True
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

    while os.path.isfile(f"train/train_conv_{image_index}.png"):
        img = cv2.imread(f"train/train_conv_{image_index}.png")
        img = np.invert(np.array([img]))
        image = img

        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()

        image = image.reshape(28, 28, 3)
        image = image.astype('float32') / 255.0
        image = np.mean(image, axis=2, keepdims=False)
        image = np.expand_dims(image, axis=0)

        pred = model.predict(image)

        print(f"This digit may be {np.argmax(pred)}")

        image_index+=1

def main():
    if CREATING_MODE:
        create_model()
    else:
        load_model()
main()