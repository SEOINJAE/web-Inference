from flask import Flask, render_template, request

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import numpy as np
import tensorflow as tf

import json
from PIL import Image
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
load = tf.saved_model.load('mnist/1')
load_inference = load.signatures["serving_default"]
app = Flask(__name__)

@app.route('/')
def load_file():
   return render_template('index.html')	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return render_template('predict.html')
@app.route('/inference', methods=['POST'])
def inference():
    image = Image.open('test_image.jpg')
    pixels = np.array(image)
    data = {'images':pixels.tolist()}
    #data=json.dumps(data)
    result = load_inference(tf.constant(data['images'], dtype=tf.float32)/255.0)
    return str(np.argmax(result['dense_1'].numpy()))

if __name__ == '__main__':
    app.run(debug=True)
