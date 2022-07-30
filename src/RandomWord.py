from urllib.request import Request, urlopen
from PickleTools import PickleTools
import random


class RandomWord:

    def __init__(self):
        self.random_word = ""
        self.random_word_list = []
        self.url = "https://svnweb.freebsd.org/csrg/share/dict/words?revision=61569&amp;view=co"

    def generate_word_list(self):
        req = Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read().decode('utf-8')
        words = webpage.split('\n')[3:-1]
        PickleTools("Datasets/TextImages/", "random_words.pkl", words).save()

    def get_word(self):
        """
        Returns a random word from the list of words.
        :return:
        """

        word_bag = PickleTools("Datasets/TextImages/", "random_words.pkl", None).load()
        self.random_word = random.choice(word_bag)
        return self.random_word
