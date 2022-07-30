import keras.backend as K


class LossFunctions:

    def __init__(self):
        self.original_secret = None
        self.original_cover = None

        self.predicted_secret = None
        self.predicted_cover = None
        self.single_beta = None

    def reveal_loss(self, s_true, s_pred):
        return self.single_beta * K.sum(K.square(s_true - s_pred))

    @staticmethod
    def cover_loss(c_true, c_pred):
        return K.sum(K.square(c_true - c_pred))

    def full_loss(self, y_true, y_pred):
        s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
        s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]

        s_loss = self.reveal_loss(s_true, s_pred)
        c_loss = self.cover_loss(c_true, c_pred)

        return sum([s_loss, c_loss])
