import numpy as np
import tensorflow as tf
from models.gan import build_generator
from models.vae import build_decoder, build_encoder
from data.preprocess import load_and_preprocess_data

def augment_data(num_samples=10000):
    (x_train, _), _ = load_and_preprocess_data()
    encoder = build_encoder(latent_dim=2)
    generator = build_generator()
    decoder = build_decoder(latent_dim=2)
    generator.load_weights('generator.h5')
    decoder.load_weights('decoder.h5')

    noise = tf.random.normal((num_samples, 100))
    generated_images = generator(noise)
    
    z_mean, z_log_var = encoder(x_train)
    z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))
    vae_generated_images = decoder(z)

    augmented_images = tf.concat([generated_images, vae_generated_images], axis=0)
    # Save or use the augmented images as needed

if __name__ == "__main__":
    augment_data()
