import pickle


class PickleTools:

    def __init__(self, path, filename, data):
        self.path = path
        self.filename = filename
        self.data = data

    def save(self):
        with open(self.path + self.filename, 'wb') as file:
            pickle.dump(self.data, file)

    def load(self):
        with open(self.path + self.filename, 'rb') as file:
            self.data = pickle.load(file)
        return self.data
