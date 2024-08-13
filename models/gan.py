import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential

def build_generator():
    model = Sequential([
        Dense(256, input_shape=(100,), activation=LeakyReLU(0.2)),
        Dense(512, activation=LeakyReLU(0.2)),
        Dense(1024, activation=LeakyReLU(0.2)),
        Dense(32 * 32 * 3, activation='tanh'),
        Reshape((32, 32, 3))
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(512, activation=LeakyReLU(0.2)),
        Dense(256, activation=LeakyReLU(0.2)),
        Dense(1, activation='sigmoid')
    ])
    return model
