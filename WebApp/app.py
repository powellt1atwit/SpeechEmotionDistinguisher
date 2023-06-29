from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import librosa as lib
import sklearn
from sklearn import preprocessing
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def main():
    if request.method == 'POST':
        
        clf = joblib.load("svm_model.pkl")

        f = request.files['audio']
        f.save(secure_filename(f.filename))
        X, samplerate = lib.load(f.filename,sr=44100,offset=0.5,duration = 3.5)

        audio_list = np.asarray(X)
  
        mel_spec = lib.feature.melspectrogram(y = audio_list, sr=44100, n_fft=2048, hop_length=512, n_mels=200)
        S_DB = lib.power_to_db(mel_spec, ref=np.max)

        spectogram_list = []

        if S_DB.shape[1] < 302:
            result = np.zeros((0,302 - S_DB.shape[1]),dtype=float)
            a = np.hstack((S_DB,result))
            spectogram_list.append(a)
        else:
            spectogram_list.append(S_DB)
        
        spectogram_array = np.asarray(spectogram_list)

        Xtemp = spectogram_array.reshape(spectogram_array.shape[0],spectogram_array.shape[1]*spectogram_array.shape[2])
        
        scaler = preprocessing.StandardScaler().fit(Xtemp)

        Xnorm = scaler.transform(Xtemp) 

        Xnormre = Xnorm.reshape(Xnorm.shape[0],spectogram_array.shape[1],spectogram_array.shape[2])

        Xnormreshapesk = Xnormre.reshape(Xnormre.shape[0],
                                                  Xnormre.shape[1]*Xnormre.shape[2])

        prediction = clf.predict(Xnormreshapesk)[0]

    else:
        prediction = ""
    
    return render_template("website.html", output = prediction)

if __name__ == '__main__':
    app.run(debug=True)