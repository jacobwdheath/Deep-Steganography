from PIL import Image
import os
import sys
from PySide2 import QtCore, QtGui, QtWidgets
from GUI.ui_main import *
import ntpath
import datetime


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set icon
        self.setWindowIcon(QtGui.QIcon("Basic Steg Testing/test_image.jpg"))

        self.ui.load_image_btn.clicked.connect(self.load_image)
        self.ui.load_image_btn.clicked.connect(self.image_info)
        self.ui.load_text_file_btn.clicked.connect(self.load_text)
        self.ui.encode_btn.clicked.connect(self.encode_message)
        self.ui.decode_btn.clicked.connect(self.decode_message)
        self.ui.encode_image_btn.clicked.connect(self.encode_image)
        self.ui.load_secret_image_btn.clicked.connect(self.load_secret_image)
        self.ui.clear_logger_btn.clicked.connect(self.clear_logger)
        self.ui.decode_image_btn.clicked.connect(self.decode_image)

        self.image_total_pixels = 0
        self.new_image = None
        self.image_path = None
        self.secret_image = None
        self.secret_image_path = None
        self.secret_image_total_pixels = 0

        self.spacing = 16
        self.ui.pixel_space_spin.setValue(self.spacing)

        self.ui.pixel_space_spin.valueChanged.connect(self.set_spacing)

        self.row_spacing = 2
        self.row_length = 8
        self.byte_width = 2

        self.text_data = ""
        self.current_chars = None
        self.prefix = None

        self.encode_decode = True

        self.show()

    def set_spacing(self):
        self.ui.pixel_space_spin.setValue(int(self.ui.pixel_space_spin.value()))

    def load_image(self):

        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open",
            "",
            "All Files (*)",
            options=options,
        )

        try:

            original_image = Image.open(image_path)
            self.image_path = image_path
            new_image = Image.new("RGB", original_image.size)
            new_image.paste(original_image)
            self.ui.image_label.setPixmap(QtGui.QPixmap(image_path))

            self.image_total_pixels = new_image.size[0] * new_image.size[1]
            self.new_image = new_image

            self.ui.logger_edit.appendPlainText(f"Loaded: {ntpath.basename(image_path)}")

        except Exception as e:
            return "Error Loading Image: " + str(e)

    def load_secret_image(self):

        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open",
            "",
            "All Files (*)",
            options=options,
        )

        try:

            original_image = Image.open(image_path)
            self.secret_image_path = image_path
            new_image = Image.new("RGB", original_image.size)
            new_image.paste(original_image)

            self.secret_image_total_pixels = new_image.size[0] * new_image.size[1]
            self.secret_image = new_image

            self.ui.logger_edit.appendPlainText(f"Loaded: {ntpath.basename(image_path)}")

        except Exception as e:
            return "Error Loading Image: " + str(e)

    def image_info(self):

        (width, height) = self.new_image.size
        image_bytes_size = os.stat(self.image_path).st_size
        image_format = os.path.splitext(self.image_path)[1]

        image_name = ntpath.basename(self.image_path)
        max_chars = self.image_total_pixels // self.spacing - 8

        creation_time = os.path.getctime(self.image_path)
        creation_time = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
        modified_time = os.path.getmtime(self.image_path)
        modified_time = datetime.datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d")

        self.ui.steg_max_chars.setText(str(max_chars))
        self.ui.steg_filename.setText(image_name)
        self.ui.steg_format.setText(image_format)
        self.ui.steg_size.setText(str(image_bytes_size))
        self.ui.steg_height.setText(str(height))
        self.ui.steg_width.setText(str(width))
        self.ui.steg_creation.setText(str(creation_time))
        self.ui.steg_modify.setText(str(modified_time))

    def load_text(self):

        options = QFileDialog.Options()
        text_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open",
            "",
            "All Files (*)",
            options=options,
        )

        try:

            self.text_data = open(text_path, "r").read()
            self.ui.logger_edit.appendPlainText(self.text_data)

        except Exception as e:
            return "Text Data Error" + str(e)

    def encode_image(self):

        cover_pixels = list(self.new_image.getdata())
        secret_pixels = list(self.secret_image.getdata())

        i = 0
        index = 0

        while i < self.image_total_pixels and index < len(secret_pixels):

            if (i % self.spacing) == 0:
                secret_pixel = secret_pixels[i]

                new_pixel = (secret_pixel[0], secret_pixel[1], secret_pixel[2])

                cover_pixels[i] = new_pixel

                index += 1
            i += 1

        new_image = Image.new("RGB", self.new_image.size)
        new_image.putdata(cover_pixels)

        options = QFileDialog.Options()
        savePath, _ = (QFileDialog.getSaveFileName(
            None,
            "Save File",
            "output.png",
            "Images (*.png *.jpg *.bmp *.tga)",
            options=options,
        ))

        new_image.save(savePath)
        self.ui.logger_edit.appendPlainText(f"Image Saved: {ntpath.basename(savePath)}")

    def decode_image(self):
        i = 0
        pixels = list(self.new_image.getdata())
        secret_pixels = []

        while i < self.image_total_pixels and i < len(pixels):
            if (i % self.spacing) == 0:
                secret_pixels += pixels[i]
            i += 1

        secret_image = Image.new("RGB", self.new_image.size)
        secret_image.putdata(secret_pixels)

        options = QFileDialog.Options()
        savePath, _ = (QFileDialog.getSaveFileName(
            None,
            "Save File",
            "output.png",
            "Images (*.png *.jpg *.bmp *.tga)",
            options=options,
        ))

        secret_image.save(savePath)

    def encode_message(self):

        self.ui.logger_edit.appendPlainText(f"Encoding Image: {ntpath.basename(self.image_path)}")
        self.prefix = str(len(self.text_data)).zfill(8)
        message = self.prefix + self.text_data
        pixels = list(self.new_image.getdata())

        i = 0
        msg_index = 0

        while i < self.image_total_pixels and msg_index < len(message):
            if (i % self.spacing) == 0:

                pixel = pixels[i]
                red = pixel[0]
                green = pixel[1]
                blue = pixel[2]

                new_red = red - (red % 35)
                new_green = green - (green % 35)
                new_blue = blue - (blue % 35)

                char = message[msg_index]
                char_id = self.id(char)

                new_red += (char_id // 3) + (char_id % 3)
                new_green += char_id // 3
                new_blue += char_id // 3

                if new_red > 255:
                    new_red -= 35
                if new_green > 255:
                    new_green -= 35
                if new_blue > 255:
                    new_blue -= 35

                new_pixel = (new_red, new_green, new_blue)
                pixels[i] = new_pixel
                msg_index += 1
            i += 1

        new_image = Image.new("RGB", self.new_image.size)
        new_image.putdata(pixels)

        options = QFileDialog.Options()
        savePath, _ = (QFileDialog.getSaveFileName(
            None,
            "Save File",
            "output.png",
            "Images (*.png *.jpg *.bmp *.tga)",
            options=options,
        ))

        new_image.save(savePath)
        self.ui.logger_edit.appendPlainText(f"Image Saved: {ntpath.basename(savePath)}")

    def decode_message(self):
        i = 0
        message = ""
        pixels = list(self.new_image.getdata())

        while i < self.image_total_pixels:
            if (i % self.spacing) == 0:
                message += self.pixelToChar(pixels[i])
            i += 1

        message_length = int(message[:8])
        chars_to_remove = -1 * (len(message) - message_length - 8)
        message_body = message[8:chars_to_remove]

        self.ui.logger_edit.appendPlainText(message_body)

    @staticmethod
    def id(char):
        asc = ord(char)

        _id = asc - 32
        if char == '\n':
            _id = 95
        elif _id < 0 or _id > 94:
            _id = 0

        return _id

    @staticmethod
    def pixelToChar(pixel):
        red = pixel[0]
        grn = pixel[1]
        blu = pixel[2]
        charId = (red % 35) + (grn % 35) + (blu % 35)
        if charId == 95:
            return '\n'
        else:
            return chr(charId + 32)

    def clear_logger(self):
        self.ui.logger_edit.clear()


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    sys.exit(app.exec_())
