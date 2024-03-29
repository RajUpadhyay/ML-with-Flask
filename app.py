import os
from flask import Flask, render_template, request
from flask import send_from_directory
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
with graph.as_default():
    # load model at very first
   set_session(sess) 
   model = load_model(STATIC_FOLDER + '/' + 'model.h5')


# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(96, 96, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255


    global sess
    global graph

    with graph.as_default():
        set_session(sess)
        predicted = model.predict(data)
        return predicted


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'NON-CANCEROUS IMAGE', 1: 'CANCEROUS IMAGE'}
        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
