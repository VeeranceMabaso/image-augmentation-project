# Image Augmentation with Deep Generative Models

## Overview
This project explores the use of Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) to enhance image classification by generating augmented training images using the CIFAR-10 dataset.

## Project Structure
- `data/`: Data loading and preprocessing scripts.
- `models/`: Definitions of GAN, VAE, and classifier models.
- `scripts/`: Scripts for training models, augmenting data, and evaluating performance.
- `prototypes/`: Prototype scripts for web interface and visualization.
- `utils/`: Utility functions for metrics and visualization.

## Setup
1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd image-augmentation-project
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Train GAN:**
    ```bash
    python scripts/train_gan.py
    ```

2. **Train VAE:**
    ```bash
    python scripts/train_vae.py
    ```

3. **Augment Data:**
    ```bash
    python scripts/augment_data.py
    ```

4. **Train Classifier:**
    ```bash
    python scripts/train_classifier.py
    ```

5. **Evaluate Classifier:**
    ```bash
    python scripts/evaluate_classifier.py
    ```

6. **Run Web Interface:**
    ```bash
    python prototypes/web_interface.py
    ```

## License
This project is licensed under the MIT License.
