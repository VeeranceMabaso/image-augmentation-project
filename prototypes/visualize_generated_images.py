import numpy as np
import matplotlib.pyplot as plt
from models.gan import build_generator

def visualize_generated_images():
    generator = build_generator()
    noise = np.random.normal(size=(16, 100))
    generated_images = generator.predict(noise)
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_generated_images()
