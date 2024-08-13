import tensorflow as tf
from models.vae import build_encoder, build_decoder
from data.preprocess import load_and_preprocess_data

def train_vae():
    (x_train, _), _ = load_and_preprocess_data()
    encoder = build_encoder(latent_dim=2)
    decoder = build_decoder(latent_dim=2)
    # VAE training code here

if __name__ == "__main__":
    train_vae()
