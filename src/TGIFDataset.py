import os
import time

import numpy as np
import random
import sys
from keras.utils.image_utils import load_img, img_to_array
from PIL import Image, GifImagePlugin
from tqdm import tqdm
from PickleTools import PickleTools
import requests
import imagesize


class TGIFDataset:

    def __init__(self):
        self.train_links_file = "Datasets/TGIF/train.txt"
        self.test_links_file = "Datasets/TGIF/test.txt"
        self.validation_links_file = "Datasets/TGIF/val.txt"

        self.train_gifs_dir = "Datasets/TGIF/train/"
        self.test_gifs_dir = "Datasets/TGIF/test/"
        self.validation_gifs_dir = "Datasets/TGIF/val/"

    def download_gifs(self):
        gif_count = 0

        with open(self.train_links_file, "rb") as file:
            gif_links = file.read().decode("utf-8").split("\n")

        gif_links = gif_links[:-1]
        gif_links = gif_links[:5000]

        for link in tqdm(gif_links):
            with open(f"{self.train_gifs_dir}{gif_count}.gif", "wb") as file:
                response = requests.get(link)
                file.write(response.content)
                gif_count += 1

            if gif_count % 10 == 0:
                time.sleep(0.5)

    def mode_gif_size(self):
        path = self.train_gifs_dir
        gif_count = 5000

        # add the width and height of every gif in the path to a list
        gif_sizes = []
        for i in tqdm(range(gif_count)):
            width, height = imagesize.get(f"{path}{i}.gif")
            gif_sizes.append((width, height))

        print(gif_sizes)

        # print the mode of the width and height values in gif_sizes
        from collections import Counter
        counter = Counter(gif_sizes)
        print(counter.most_common(1))


if __name__ == "__main__":
    tgif = TGIFDataset()
    tgif.mode_gif_size()
    # 6GB 5000 GIFS
    # tgif.download_gifs()
