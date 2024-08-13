from scripts.train_gan import train_gan
from scripts.train_vae import train_vae
from scripts.augment_data import augment_data
from scripts.train_classifier import train_classifier
from scripts.evaluate_classifier import evaluate_classifier

def main():
    # Train GAN
    train_gan()

    # Train VAE
    train_vae()

    # Augment data
    augment_data()

    # Train classifier
    train_classifier()

    # Evaluate classifier
    evaluate_classifier()

if __name__ == "__main__":
    main()
