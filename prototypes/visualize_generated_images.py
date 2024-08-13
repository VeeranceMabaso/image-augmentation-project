import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import build_generator

def visualize_generated_images():
    generator = build_generator()
    generator.load_weights('generator.h5')
    
    noise = np.random.normal(size=(16, 100))
    generated_images = generator.predict(noise)
    
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((generated_images[i] + 1.0) / 2.0)  # Rescale to [0, 1]
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_generated_images()
