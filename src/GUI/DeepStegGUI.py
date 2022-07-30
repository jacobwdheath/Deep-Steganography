from PIL import Image
import os
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtGui import QImage

from GUI.ai_main import *
import ntpath
import datetime
from keras.utils.image_utils import load_img, img_to_array
from keras.models import load_model
from PIL import Image, ImageQt
import numpy as np
import keras.backend as K
from src.TinyDataset import TinyImageDataset
import matplotlib.pyplot as plt
from random import getstate


sys.modules['Image'] = Image
beta = 0.25


def reveal_loss(s_true, s_pred):
    return beta * K.sum(K.square(s_true - s_pred))


def cover_loss(c_true, c_pred):
    return K.sum(K.square(c_true - c_pred))


def full_loss(self, y_true, y_pred):
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]

    s_loss = self.reveal_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))

    return sum([s_loss, c_loss])


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # set icon
        self.setWindowIcon(QtGui.QIcon("Basic Steg Testing/test_image.jpg"))

        # Model load here

        # encoder_model = tf.keras.load_models("modeldata/encoder_model.h5")
        # decoder_model = tf.keras.load_models("modeldata/decoder_model.h5")

        self.autoencoder_model = load_model(r"C:\Users\Jacob\Documents\GitHub\JH-Deep-Steganography\src\Image "
                                            r"Models\Saved Models\Single Image\autoencoder\autoencoder.h5",
                                            custom_objects={"full_loss": full_loss})

        self.ui.load_secret_image_btn.clicked.connect(self.load_secret_image)
        self.ui.load_cover_image_btn.clicked.connect(self.load_cover_image)
        self.ui.encode_cover_btn.clicked.connect(self.encode_cover_image)
        self.ui.decode_secret_btn.clicked.connect(self.decode_cover_image)
        self.ui.clear_logger_btn.clicked.connect(self.clear_logger)
        self.ui.save_steg_img_btn.clicked.connect(self.save_steg_image)

        self.secret_image_path = r"C:\Users\Jacob\Documents\GitHub\JH-Deep-Steganography" \
                                 r"\src\GUI\test-data\tinyimage\test_0.JPEG"
        self.cover_image_path = r"C:\Users\Jacob\Documents\GitHub\JH-Deep-Steganography" \
                                r"\src\GUI\test-data\tinyimage\test_1.JPEG"
        self.encoded_cover_path = None
        self.decoded_secret_path = None

        self.embedded_cover = None

        img_s = load_img(self.secret_image_path)
        img_c = load_img(self.cover_image_path)
        s = img_to_array(img_s)
        c = img_to_array(img_c)

        c = c / 255.0
        s = s / 255.0

        c = c.reshape(1, *c.shape)
        s = s.reshape(1, *s.shape)

        self.autoencoder_model.predict([c, s])

        self.secret_image_path = None
        self.cover_image_path = None
        self.encoded_cover_path = None
        self.decoded_secret_path = None

        self.secret_image = None
        self.cover_image = None
        self.encoded_cover = None
        self.decoded_secret = None

        self.show()

    def load_secret_image(self):

        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open",
            "",
            "Images (*.png *.jpg *.jpeg)",
            options=options,
        )

        try:

            original_image = Image.open(image_path)
            self.secret_image_path = image_path
            new_image = Image.new("RGB", original_image.size)
            new_image.paste(original_image)
            self.secret_image = new_image

            self.ui.secret_image.setPixmap(QtGui.QPixmap(self.secret_image_path))

            self.ui.logger_edit.appendPlainText(f"Loaded: {ntpath.basename(image_path)}")

        except Exception as e:
            return "Error Loading Image: " + str(e)

    def load_cover_image(self):

        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open",
            "",
            "Images (*.png *.jpg *.jpeg)",
            options=options,
        )

        try:

            original_image = Image.open(image_path)
            self.cover_image_path = image_path
            new_image = Image.new("RGB", original_image.size)
            new_image.paste(original_image)
            self.cover_image = new_image

            self.ui.cover_image.setPixmap(QtGui.QPixmap(self.cover_image_path))

            self.ui.logger_edit.appendPlainText(f"Loaded: {ntpath.basename(image_path)}")

        except Exception as e:
            return "Error Loading Image: " + str(e)

        except Exception as e:
            return "Error Loading Image: " + str(e)

    def encode_cover_image(self):

        img_s = load_img(self.secret_image_path)
        img_c = load_img(self.cover_image_path)
        s = img_to_array(img_s)
        c = img_to_array(img_c)

        c = c / 255.0
        s = s / 255.0

        c = c.reshape(1, *c.shape)
        s = s.reshape(1, *s.shape)

        decoded = self.autoencoder_model.predict([s, c])
        decoded_S1, decoded_cover_image = decoded[..., 0:3], decoded[..., 3:6]
        decoded_S1 = decoded_S1.reshape(decoded_S1.shape[1:])
        decoded_cover_image = decoded_cover_image.reshape(decoded_cover_image.shape[1:])
        self.decoded_secret = decoded_S1

        embedded_cover = self.format_image(decoded_cover_image)
        self.embedded_cover = embedded_cover
        self.ui.encoded_cover_label.setPixmap(embedded_cover)

    def decode_cover_image(self):
        decoded_secret = self.format_image(self.decoded_secret)
        self.ui.decoded_secret_image.setPixmap(decoded_secret)

    @staticmethod
    def format_image(image_array):
        image = np.clip(image_array, 0, 1)
        image = Image.fromarray((image * 255).astype(np.uint8))
        qt_image = ImageQt.ImageQt(image)
        qt_image.convertToFormat(QImage.Format_ARGB32)
        pix = QtGui.QPixmap.fromImage(qt_image)
        return pix

    def image_info(self, image_path):
        (width, height) = self.new_image.size
        image_bytes_size = os.stat(self.image_path).st_size
        image_format = os.path.splitext(self.image_path)[1]

        image_name = ntpath.basename(self.image_path)
        max_chars = self.image_total_pixels // self.spacing - 8

        creation_time = os.path.getctime(self.image_path)
        creation_time = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
        modified_time = os.path.getmtime(self.image_path)
        modified_time = datetime.datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d")

        self.ui.logger_edit.appendPlainText(f"Image Saved: {ntpath.basename(image_path)}")

    def clear_logger(self):
        self.ui.logger_edit.clear()

    def save_steg_image(self):
        """options = QFileDialog.Options()
        image_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save",
            "",
            "Images (*.png *.jpg *.jpeg)",
            options=options,
        )"""

        path = r"C:\Users\Jacob\Documents\GitHub\StegExpose\images"

        if path:
            self.embedded_cover.save(path + r"\embedded_cover.png")

        # print out the random seed used in this execution from random module

        print(getstate())


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    sys.exit(app.exec_())
