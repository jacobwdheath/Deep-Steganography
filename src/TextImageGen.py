import os
import sys
from PIL import Image, ImageDraw, ImageFont
import random
from RandomWord import RandomWord


class TextImageGen:

    def __init__(self, width, height, message, text_size):
        self.save_path = "Datasets/TextImages/train/"

        self.width = width
        self.height = height
        self.message = message
        self.font = ImageFont.truetype("arial.ttf", size=text_size)

        self.generate_text_image()

    def generate_text_image(self):

        bg_colours = ["red", "green", "blue"]

        bg_colour = random.choice(bg_colours)

        text_colour = ()

        if bg_colour == "red":
            text_colour = (0, 255, 0)
        elif bg_colour == "green":
            text_colour = (255, 0, 0)
        elif bg_colour == "blue":
            text_colour = (255, 165, 0)

        image = Image.new("RGB", (self.width, self.height), color=bg_colour)

        draw_image = ImageDraw.Draw(image)

        text_width, text_height = draw_image.textsize(self.message, font=self.font)
        text_x = (self.width - text_width) / 2
        text_y = (self.height - text_height) / 2
        draw_image.text((text_x, text_y), self.message, font=self.font, fill=text_colour)

        file_title = self.message.replace("\n", "_")

        image.save(f"{self.save_path}{bg_colour}_{file_title}.png")


if __name__ == "__main__":

    num_images = 0

    for i in range(num_images):
        random_word1 = RandomWord().get_word()
        random_word2 = RandomWord().get_word()

        message = f"{random_word1}\n{random_word2}"

        TextImageGen(width=64, height=64, message=message, text_size=10)
