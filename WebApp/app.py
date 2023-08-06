from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import librosa as lib
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def home():
    if request.method == 'POST':
        
        #Import model and import user input
        clf = joblib.load("model.pkl")

        f = request.files['audio']
        f.save(secure_filename(f.filename))
        data, sample_rate = lib.load(f.filename)
        
        #Run preprocessing on user data
        def extract_features(data):
            # ZCR
            result = np.array([])
            zcr = np.mean(lib.feature.zero_crossing_rate(y=data).T, axis=0)
            result=np.hstack((result, zcr)) # stacking horizontally

            # Chroma_stft
            stft = np.abs(lib.stft(data))
            chroma_stft = np.mean(lib.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_stft)) # stacking horizontally

            # MFCC
            mfcc = np.mean(lib.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mfcc)) # stacking horizontally

            # Root Mean Square Value
            rms = np.mean(lib.feature.rms(y=data).T, axis=0)
            result = np.hstack((result, rms)) # stacking horizontally

            # MelSpectogram
            mel = np.mean(lib.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel)) # stacking horizontally
    
            return result

        def get_features(path):
            # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
            data, sample_rate = lib.load(path, duration=2.5, offset=0.6)
    
            # without augmentation
            res1 = extract_features(data)
            result = np.array(res1)
    
            return result
        
        X = []
        path = f.filename
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
        
        Features = pd.DataFrame(X)
        X = Features.T.values
        print(X.shape)

        scaler = joblib.load("scaler.pkl")
        X = scaler.transform(X)

        X = np.expand_dims(X, axis=2)

        #Run model to predict on user input and display prediction classification
        pred = clf.predict(X)[0]
        print(pred.shape)

        encoder = joblib.load("encoder.pkl")
        enc = encoder.inverse_transform(pred.reshape(1,-1))

        prediction = enc.flatten()[0].capitalize()

    else:
        prediction = ""
    
    return render_template("website.html", output = prediction)

@app.route('/tips/')

def tips():
    return render_template("tips.html")

@app.route('/about/')

def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)