from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_image():
    # Use your GAN or VAE to generate an image
    img = np.random.rand(32, 32, 3)  # Placeholder
    buf = BytesIO()
    plt.imsave(buf, img)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return render_template('image.html', img_data=img_str)

if __name__ == "__main__":
    app.run(debug=True)
