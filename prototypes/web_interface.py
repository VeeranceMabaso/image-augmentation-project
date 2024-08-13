from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gan import build_generator

app = Flask(__name__)

generator = build_generator()
generator.load_weights('generator.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_image():
    noise = np.random.normal(size=(1, 100))
    img = generator.predict(noise)[0]
    img = (img + 1.0) / 2.0  # Rescale to [0, 1]
    buf = BytesIO()
    plt.imsave(buf, img)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return render_template('image.html', img_data=img_str)

if __name__ == "__main__":
    app.run(debug=True)
