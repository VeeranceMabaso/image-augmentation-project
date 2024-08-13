import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import build_encoder, build_decoder
from data.preprocess import load_and_preprocess_data
import warnings
warnings.filterwarnings("ignore")

def vae_loss(x_true, x_pred, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_true, x_pred))
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    return reconstruction_loss + kl_loss

def train_vae(epochs=50, batch_size=64):
    (x_train, _), _ = load_and_preprocess_data()
    x_train = x_train * 2.0 - 1.0  # Rescale to [-1, 1]
    
    latent_dim = 2
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    
    optimizer = tf.keras.optimizers.Adam()
    
    for epoch in range(epochs):
        for batch_start in range(0, len(x_train), batch_size):
            x_batch = x_train[batch_start:batch_start + batch_size]
            with tf.GradientTape() as tape:
                z_mean, z_log_var = encoder(x_batch)
                z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))
                x_pred = decoder(z)
                loss = vae_loss(x_batch, x_pred, z_mean, z_log_var)
            gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.numpy()}")

    encoder.save('encoder.h5')
    decoder.save('decoder.h5')

if __name__ == "__main__":
    train_vae()
