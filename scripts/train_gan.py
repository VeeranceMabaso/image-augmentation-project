import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import build_encoder, build_decoder
from models.gan import build_generator, build_discriminator
from data.preprocess import load_and_preprocess_data
import warnings
warnings.filterwarnings("ignore")

def train_gan(epochs=100, batch_size=64):
    (x_train, _), _ = load_and_preprocess_data()
    x_train = x_train * 2.0 - 1.0  # Rescale to [-1, 1]
    
    generator = build_generator()
    discriminator = build_discriminator()
    
    gan = tf.keras.Sequential([generator, discriminator])
    
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    
    for epoch in range(epochs):
        # Train discriminator
        idx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=x_train.shape[0], dtype=tf.int32)
        real_imgs = tf.gather(x_train, idx)
        fake_imgs = generator(tf.random.normal((batch_size, 100)))
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
        
        # Train generator
        g_loss = gan.train_on_batch(tf.random.normal((batch_size, 100)), real_labels)
        
        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
    
    generator.save('generator.h5')

if __name__ == "__main__":
    train_gan()
