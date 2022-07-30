import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.utils import plot_model
import numpy as np
import pickle
import os


class CNNModels:

    def __init__(self):
        self.models = ["one_secret", "three_secret"]

    def __str__(self):
        return self.models

    def list_models(self):
        print(self.models)

    @staticmethod
    def one_secret_encoder(secret_input, activation, filter1, filter2, filter3):
        secret_input1 = Input(shape=secret_input)
        cover_input = Input(shape=secret_input)

        # Preparation Network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_3x3_1')(
            secret_input1)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_4x4_1')(
            secret_input1)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_5x5_1')(
            secret_input1)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_5x5_1')(x)
        x_row1 = concatenate([x3, x4, x5])

        x = concatenate([cover_input, x_row1])

        # Hiding network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='convo_encodode0_3x3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='convo_encodode0_4x4')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='convo_encodode0_5x5')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='convo_encodode1_3x3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='convo_encodode1_4x4')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='convo_encodode1_5x5')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='convo_encodode2_3x3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='convo_encodode2_4x4')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='convo_encodode2_5x5')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='convo_encodode3_3x3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='convo_encodode3_4x4')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='convo_encodode3_5x5')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='convo_encodode4_3x3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='convo_encodode4_4x4')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='convo_encodode5_5x5')(
            x)
        x = concatenate([x3, x4, x5])

        output_Cerror_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_C')(
            x)

        return Model(inputs=[secret_input1, cover_input],
                     outputs=output_Cerror_term,
                     name='Encoder')

    @staticmethod
    def three_secret_encoder(secret_input, activation, filter1, filter2, filter3):
        secret_input1 = Input(shape=secret_input)
        secret_input2 = Input(shape=secret_input)
        secret_input3 = Input(shape=secret_input)
        cover_input = Input(shape=secret_input)

        # Preparation Network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_3x3_1')(
            secret_input1)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_4x4_1')(
            secret_input1)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_5x5_1')(
            secret_input1)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_5x5_1')(x)
        x_row1 = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_3x3_2')(
            secret_input2)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_4x4_2')(
            secret_input2)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_5x5_2')(
            secret_input2)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_5x5_2')(x)
        x2 = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_3x3_3')(
            secret_input3)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_4x4_3')(
            secret_input3)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress0_5x5_3')(
            secret_input3)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='conv_compress1_5x5_3')(x)
        x3__1 = concatenate([x3, x4, x5])

        x = concatenate([cover_input, x_row1, x2, x3__1])

        # --------------------------------------------------------------------------------------------------------------
        # Hiding network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee0_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee0_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee0_5x5_1')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee1_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee1_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee1_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee2_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee2_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee2_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee3_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee3_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee3_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee4_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee4_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_encododee4_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        output_Cerror_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_C')(
            x)

        return Model(inputs=[secret_input1, secret_input2, secret_input3, cover_input],
                     outputs=output_Cerror_term,
                     name='Encoder')

    @staticmethod
    def one_secret_decoder(secret_input, activation, filter1, filter2, filter3):
        # Reveal network
        reveal_input = Input(shape=secret_input)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_3x3_1')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_4x4_1')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_5x5_1')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr5_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        output_S1error_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation,
                                     name='output_S1')(x)

        return Model(inputs=reveal_input, outputs=output_S1error_term)

    @staticmethod
    def three_secret_decoder_1(secret_input, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=secret_input)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_3x3_1')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_4x4_1')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_5x5_1')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr5_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        output_S1error_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation,
                                     name='output_S1')(x)

        return Model(inputs=reveal_input, outputs=output_S1error_term)

    @staticmethod
    def three_secret_decoder_2(secret_input, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=secret_input)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise2')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_3x3_2')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_4x4_2')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_5x5_2')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr5_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        output_S2error_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation,
                                     name='output_S2')(x)

        return Model(inputs=reveal_input, outputs=output_S2error_term)

    @staticmethod
    def three_secret_decoder_3(secret_input, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=secret_input)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_3x3_3')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_4x4_3')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr0_5x5_3')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr3_5x5_3')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr4_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation,
                    name='convo_decooderr5_5x5_3')(x)
        x = concatenate([x3, x4, x5])

        output_S3error_term = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation,
                                     name='output_S3')(x)

        return Model(inputs=reveal_input, outputs=output_S3error_term)

    def one_secret_build_networks(self, secret_input, reveal_loss, full_loss, activation, filter1, filter2,
                                  filter3):
        secret_input1 = Input(shape=secret_input)
        cover_input = Input(shape=secret_input)

        encoder = self.one_secret_encoder(secret_input, activation, filter1, filter2, filter3)

        decoder = self.one_secret_decoder(secret_input, activation, filter1, filter2, filter3)
        decoder.compile(optimizer="adam", loss=reveal_loss)
        decoder.trainable = False

        # Encoded Image
        output_Cerror_term = encoder([secret_input1, cover_input])
        # Decoded Image
        output_S1error_term = decoder(output_Cerror_term)

        autoencoder = Model(inputs=[secret_input1, cover_input],
                            outputs=concatenate([output_S1error_term, output_Cerror_term]))
        autoencoder.compile(optimizer='adam', loss=full_loss)

        return encoder, decoder, autoencoder

    def three_secret_build_networks(self, secret_input, reveal_loss, full_loss, activation, filter1, filter2,
                                    filter3):
        secret_input_1 = Input(shape=secret_input)
        secret_input_2 = Input(shape=secret_input)
        secret_input_3 = Input(shape=secret_input)
        cover_input = Input(shape=secret_input)

        encoder = self.three_secret_encoder(secret_input, activation, filter1, filter2, filter3)

        decoder1 = self.three_secret_decoder_1(secret_input, activation, filter1, filter2, filter3)
        decoder1.compile(optimizer="adam", loss=reveal_loss)
        decoder1.trainable = False

        decoder2 = self.three_secret_decoder_2(secret_input, activation, filter1, filter2, filter3)
        decoder2.compile(optimizer="adam", loss=reveal_loss)
        decoder2.trainable = False

        decoder3 = self.three_secret_decoder_3(secret_input, activation, filter1, filter2, filter3)
        decoder3.compile(optimizer="adam", loss=reveal_loss)
        decoder3.trainable = False

        output_Cerror_term = encoder([secret_input_1, secret_input_2, secret_input_3, cover_input])
        output_S1error_term = decoder1(output_Cerror_term)
        output_S2error_term = decoder2(output_Cerror_term)
        output_S3error_term = decoder3(output_Cerror_term)

        autoencoder = Model(inputs=[secret_input_1, secret_input_2, secret_input_3, cover_input],
                            outputs=concatenate(
                                [output_S1error_term, output_S2error_term, output_S3error_term, output_Cerror_term]))
        autoencoder.compile(optimizer='adam', loss=full_loss)

        return encoder, decoder1, decoder2, decoder3, autoencoder

    @staticmethod
    def learning_rate(epoch_number):
        return 0.0001

    def train_one_secret(self, batch_size, epochs, path, shape, reveal_loss, full_loss, secret_input,
                         cover_input, verbose, save_interval, activation, filter1, filter2, filter3):

        """

        :param filter3:
        :param filter2:
        :param filter1:
        :param activation:
        :param batch_size:
        :param epochs: number of epochs
        :param path: weights model and loss history save path
        :param shape: input shape
        :param reveal_loss: loss for network prep and hide
        :param full_loss: loss for autoencoder
        :param secret_input: secret image input
        :param cover_input: cover image input
        :param verbose: log in console or not (0, 1)
        :param save_interval: save every n epochs
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)

        encoder_model, decoder_model, autoencoder_model = self.one_secret_build_networks(

            shape, reveal_loss, full_loss, activation, filter1, filter2, filter3

        )

        history = []
        decode_loss = []
        auto_encode_loss = []
        size = secret_input.shape[0]

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            np.random.shuffle(secret_input)
            np.random.shuffle(cover_input)

            for i in range(0, size, batch_size):
                secret_batch = secret_input[i: min(i + batch_size, size)]
                cover_batch = cover_input[i: min(i + batch_size, size)]

                # hide network loss error term
                cover_loss = encoder_model.predict([secret_batch, cover_batch], verbose=verbose)

                auto_encode_loss.append(autoencoder_model.train_on_batch(x=[secret_batch, cover_batch],
                                                                         y=np.concatenate((secret_batch, cover_batch),
                                                                                          axis=3)))
                # reveal loss
                decode_loss.append(decoder_model.train_on_batch(x=cover_loss, y=secret_batch))
                K.set_value(autoencoder_model.optimizer.lr, self.learning_rate(epoch))
                K.set_value(decoder_model.optimizer.lr, self.learning_rate(epoch))

            if (epoch + 1 % (epochs / 4)) == 0:
                encoder_model.save_weights(path + "encoder " + str(epoch))
                decoder_model.save_weights(path + "decoder " + str(epoch))
                autoencoder_model.save_weights(path + "autoencoder" + str(epoch))

            history.append(np.mean(auto_encode_loss))

        def save_files():

            with open(path + "autoencoder_loss.pckl", 'wb') as f:
                pickle.dump(auto_encode_loss, f)

            with open(path + "decode_loss.pckl", 'wb') as f:
                pickle.dump(decode_loss, f)

            with open(path + "loss_history.pckl", 'wb') as f:
                pickle.dump(history, f)

            encoder_model.save(path + "encoder.h5")
            decoder_model.save(path + "decoder.h5")
            autoencoder_model.save(path + "autoencoder.h5")

        save_files()

        return encoder_model, decoder_model, autoencoder_model

    def train_three_secret(self,
                           batch_size, epochs, path, shape,
                           reveal_loss, full_loss, secret1_input,
                           secret2_input, secret3_input,
                           cover_input, verbose, save_interval,
                           activation, filter1, filter2, filter3
                           ):

        """

        :param secret1_input:
        :param secret2_input:
        :param secret3_input:
        :param filter3:
        :param filter2:
        :param filter1:
        :param activation:
        :param batch_size:
        :param epochs: number of epochs
        :param path: weights model and loss history save path
        :param shape: input shape
        :param reveal_loss: loss for network prep and hide
        :param full_loss: loss for autoencoder
        :param cover_input: cover image input
        :param verbose: log in console or not (0, 1)
        :param save_interval: save every n epochs
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)

        encoder_model, decoder1_model, decoder2_model, decoder3_model, autoencoder_model = \
            self.three_secret_build_networks(shape, reveal_loss, full_loss, activation,
                                             filter1, filter2, filter3)

        history = []
        decode1_loss = []
        decode2_loss = []
        decode3_loss = []
        auto_encode_loss = []
        size = secret1_input.shape[0]

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            np.random.shuffle(secret1_input)
            np.random.shuffle(secret2_input)
            np.random.shuffle(secret3_input)
            np.random.shuffle(cover_input)

            for i in range(0, size, batch_size):
                secret1_batch = secret1_input[i: min(i + batch_size, size)]
                secret2_batch = secret2_input[i: min(i + batch_size, size)]
                secret3_batch = secret3_input[i: min(i + batch_size, size)]
                cover_batch = cover_input[i: min(i + batch_size, size)]

                # hide network loss error term
                cover_loss = encoder_model.predict([secret1_batch, secret2_batch, secret3_batch, cover_batch],
                                                   verbose=verbose)

                auto_encode_loss.append(
                    autoencoder_model.train_on_batch(x=[secret1_batch, secret2_batch, secret3_batch, cover_batch],
                                                     y=np.concatenate(
                                                         (secret1_batch, secret2_batch, secret3_batch, cover_batch),
                                                         axis=3)))
                # reveal loss
                decode1_loss.append(decoder1_model.train_on_batch(x=cover_loss, y=secret1_batch))
                decode2_loss.append(decoder2_model.train_on_batch(x=cover_loss, y=secret2_batch))
                decode3_loss.append(decoder3_model.train_on_batch(x=cover_loss, y=secret3_batch))

                K.set_value(autoencoder_model.optimizer.lr, self.learning_rate(epoch))
                K.set_value(decoder1_model.optimizer.lr, self.learning_rate(epoch))
                K.set_value(decoder2_model.optimizer.lr, self.learning_rate(epoch))
                K.set_value(decoder3_model.optimizer.lr, self.learning_rate(epoch))

            if (epoch + 1 % (epochs / 4)) == 0:
                encoder_model.save_weights(path + "encoder " + str(epoch))
                decoder1_model.save_weights(path + "decoder1 " + str(epoch))
                decoder2_model.save_weights(path + "decoder2 " + str(epoch))
                decoder3_model.save_weights(path + "decoder3 " + str(epoch))
                autoencoder_model.save_weights(path + "autoencoder" + str(epoch))

            history.append(np.mean(auto_encode_loss))

        def save_files():

            with open(path + "autoencoder_loss.pckl", 'wb') as f:
                pickle.dump(auto_encode_loss, f)

            with open(path + "decode_loss1.pckl", 'wb') as f:
                pickle.dump(decode1_loss, f)

            with open(path + "decode_loss2.pckl", 'wb') as f:
                pickle.dump(decode2_loss, f)

            with open(path + "decode_loss3.pckl", 'wb') as f:
                pickle.dump(decode3_loss, f)

            with open(path + "loss_history.pckl", 'wb') as f:
                pickle.dump(history, f)

            encoder_model.save(path + "encoder.h5")
            decoder1_model.save(path + "decoder1.h5")
            decoder2_model.save(path + "decoder2.h5")
            decoder3_model.save(path + "decoder3.h5")
            autoencoder_model.save(path + "autoencoder.h5")

        save_files()

        return encoder_model, decoder1_model, decoder2_model, decoder3_model, autoencoder_model
