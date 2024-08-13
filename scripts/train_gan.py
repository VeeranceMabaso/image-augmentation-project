import tensorflow as tf
from models.gan import build_generator, build_discriminator
from data.preprocess import load_and_preprocess_data

def train_gan():
    (x_train, _), _ = load_and_preprocess_data()
    generator = build_generator()
    discriminator = build_discriminator()
    # GAN training code here

if __name__ == "__main__":
    train_gan()
