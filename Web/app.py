from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
import pandas as pd
# from pushbullet import PushBullet
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

#Import the required modules
import keras
import keras.utils as image
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

############################################# Genre-Poster FUNCTIONS ################################################

# Function to predict the genre for a particular movie
def find_genre(test_path):
    a=""
    model = load_model("models/Model_4d.h5") 
    #model = load_model("models/Model_6c.h5") 
    img = image.load_img(test_path,target_size=(200,150,3))
    img = image.img_to_array(img)
    img = img/255
    prob = model.predict(img.reshape(1,200,150,3))
    top_3 = np.argsort(prob[0])[:-3:-1]

    column_lookups = pd.read_csv("data/Encoded_data_column_lookup.csv", delimiter=" ")
    classes = np.asarray(column_lookups.iloc[0:5, 0])

    a += "{} ({:.3}) & ".format(classes[top_3[0]], prob[0][top_3[0]])
    a += "{} ({:.3})".format(classes[top_3[1]], prob[0][top_3[1]])
    return a

############################################# Genre-Synopsis FUNCTIONS ################################################
# Function to predict the genre for a particular movie
def find_genre_synopsis(synop):
    a = np.array([synop], dtype=object)

    # Load the saved vectorizer from a file
    with open('models/vectorizer_synopsis.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    ft = vectorizer.transform(a)

    svm_model = pickle.load(open("models/model_svm.sav", 'rb'))
    pred = svm_model.predict(ft)

    # mnb_model = pickle.load(open("models/model_mnb.sav", 'rb'))
    # pred = mnb_model.predict(ft)

    return pred[0]

############################################# Review-Rating FUNCTIONS ################################################
# Function to rating for a movie with the review
def get_rating(review):
    
    # Create the Pandas Series
    series = pd.Series(review)
    model = pickle.load(open("models/model_mnb.pk", 'rb'))
    
    #model = pickle.load(open("models/model_mnb.pk", 'rb'))
    #model = pickle.load(open("models/model_mnb.pk", 'rb'))
    #model = pickle.load(open("models/model_mnb.pk", 'rb'))

    pred = model.predict(series)

    return pred[0]

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/posters')
def posters():
    return render_template('posters.html')

@app.route('/synopsis')
def synopsis():
    return render_template('synopsis.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

########################### Result Functions ########################################


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        moviename = request.form['firstname']
        movieid = request.form['lastname']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img_path = 'static/uploads/'+ filename
            pred = find_genre(img_path)
            return render_template('resultp.html', filename=filename, fn=moviename, ln=movieid, r=pred)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        moviename = request.form['firstname']
        movieid = request.form['lastname']
        synopsis = request.form['plot']
        pred = find_genre_synopsis(synopsis)
        return render_template('results.html', fn=moviename, ln=movieid, r=pred)

@app.route('/resultr', methods=['GET', 'POST'])
def resultr():
    if request.method == 'POST':
        print(request.url)
        moviename = request.form['firstname']
        movieid = request.form['lastname']
        review = request.form['review']
        pred=get_rating(review)
        return render_template('resultr.html', fn=moviename, ln=movieid, r=pred)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
