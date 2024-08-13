import numpy as np
from PIL import Image
import os
import tensorflow as tf

def prepare_data_paths():
    augmented_dir = 'augmented_images/'
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

def load_and_preprocess_data():
    prepare_data_paths()
    
    # Load original data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize original data to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Initialize lists for augmented data
    augmented_images = []
    augmented_labels = []
    augmented_dir = 'augmented_images/'

    # Load augmented data if directory exists and has files
    if os.listdir(augmented_dir):
        for filename in os.listdir(augmented_dir):
            img = Image.open(os.path.join(augmented_dir, filename))
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            augmented_images.append(img)
            # Example: Random label assignment for demonstration
            augmented_labels.append(np.random.randint(0, 10))  # Random label assignment
    else:
        print("No augmented images found. Please run the augmentation script to generate data.")

    if augmented_images:
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        # Combine original and augmented data
        x_train_combined = np.concatenate([x_train, augmented_images], axis=0)
        y_train_combined = np.concatenate([y_train, augmented_labels], axis=0)
    else:
        x_train_combined = x_train
        y_train_combined = y_train

    return (x_train_combined, y_train_combined), (x_test, y_test)
