import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)
