import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    augmented_images = (augmented_images * 255).numpy().astype(np.uint8)  # Convert to valid image format
    
    # Save augmented images
    for i, img in enumerate(augmented_images):
        img = Image.fromarray(img)
        img.save(f'augmented_images/image_{i}.png')

if __name__ == "__main__":
    augment_data()
