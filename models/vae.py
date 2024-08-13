import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

def build_encoder(latent_dim):
    inputs = Input(shape=(32, 32, 3))
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    return Model(inputs, [z_mean, z_log_var])

def build_decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(512, activation='relu')(latent_inputs)
    x = Dense(32 * 32 * 3, activation='sigmoid')(x)
    outputs = Reshape((32, 32, 3))(x)
    return Model(latent_inputs, outputs)
