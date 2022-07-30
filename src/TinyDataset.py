# Methods to help with training and Data manipulation.
import os
from keras.utils.image_utils import load_img, img_to_array
import random
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import sys
from PIL import Image

sys.modules['Image'] = Image


class TinyImageDataset:

    def __init__(self, path: str, num_classes: int, normalize: bool):

        """

        :param path:
        :param num_classes:
        :param normalize:

        Use load_data() to return a numpy array of image data.
        Use show_examples() to show a random selection of images.

        """

        self.path = None
        self.num_classes = None
        self.normalize = None

        valid_kwargs = {'path': path, 'num_classes': num_classes, 'normalize': normalize}
        for key, value in valid_kwargs.items():
            if key not in valid_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
            setattr(self, key, value)

    def load_data(self):
        X_train = []

        for i in os.listdir(self.path):
            _dir = os.path.join(self.path, i, 'images')
            imgs = os.listdir(_dir)
            random.shuffle(imgs)

            for img_name in imgs[0:self.num_classes]:
                img_i = load_img(os.path.join(_dir, img_name))
                x = img_to_array(img_i)
                X_train.append(x)

        random.shuffle(X_train)

        # Numpy array
        if self.normalize:
            X_train = array(X_train) / 255.0
            return X_train

        else:
            return array(X_train)

    @staticmethod
    def show_examples(X_train, columns, rows, figsize):
        fig = plt.figure(figsize=figsize)

        for i in range(1, columns * rows + 1):
            random_index = np.random.choice(X_train.shape[0])
            fig.add_subplot(rows, columns, i)
            plt.imshow(X_train[random_index])
            plt.axis('off')

        plt.show()
