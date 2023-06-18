from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.python.client import device_lib


def check_GPU():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device_lib.list_local_devices()


def preprocess_images(path):
    image_files = os.listdir(path)

    for image_file in image_files:
        print(f"{image_file} processing...")

        image_path = os.path.join(path, image_file)
        image = cv2.imread(image_path)

        new_size = (1500, 1000)
        resized_image = cv2.resize(image, new_size)
        new_image_file = os.path.join(path, image_file)

        cv2.imwrite(new_image_file, resized_image)
        # cv2.imwrite(path, resized_image)


def show(image: np.ndarray, title="Image", cmap_type="gray", axis=False):
    """
    A function to display np.ndarrays as images
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    if not axis:
        plt.axis("off")
    plt.margins(0, 0)
    plt.show()


if __name__ == '__main__':
    check_GPU()
