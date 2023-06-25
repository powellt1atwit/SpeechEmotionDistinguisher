from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import librosa as lib
import sklearn
from sklearn import preprocessing

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def main():
    if request.method == "POST":
        clf = joblib.load("svm_model.pkl")

        audio = request.form.get("audio")

        X = pd.DataFrame([[audio]], columns=["audio"])

        a_list = [1, 2]
        distribution = [0.0, 1.0]

        manipulated_audio = []

        random_number = np.random.choice(a_list, p = distribution)
        if random_number == 2:
            data = manipulate_noise(X,0.008)
            manipulated_audio.append(data)
        
        a_list = [1, 2]
        distribution = [0.0, 1.0]

        manipulated_audio_pitch = []

        random_number = np.random.choice(a_list, p = distribution)
        if random_number == 2:
            data = manipulate_pitch(audio)
            manipulated_audio_pitch.append(data)
        
        X2 = np.concatenate((X,np.asarray(manipulated_audio),
                               np.asarray(manipulated_audio_pitch)),axis = 0)
        
        mel_spec = lib.feature.melspectrogram(y = X[0], sr=44100, n_fft=2048, hop_length=512, n_mels=200)
        S_DB = lib.power_to_db(mel_spec, ref=np.max)

        spectogram_list = []

        if S_DB.shape[1] < 0:
            result = np.zeros((0,0 - S_DB.shape[1]),dtype=float)
            a = np.hstack((S_DB,result))
            spectogram_list.append(a)
        else:
            spectogram_list.append(S_DB)
        
        spectogram_array = np.asarray(spectogram_list)

        Xtemp = spectogram_array.reshape(spectogram_array.shape[0],spectogram_array.shape[1]*spectogram_array.shape[2])
        
        scaler = preprocessing.StandardScaler().fit(Xtemp)

        Xnorm = scaler.transform(Xtemp) 

        Xnormre = Xnorm.reshape(Xnorm.shape[0],spectogram_array.shape[1],spectogram_array.shape[2])

        prediction = clf.predict(Xnormre)[0]

    else:
        prediction = ""
    
    return render_template("website.html", output = prediction)

def manipulate_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def manipulate_pitch(data):
    return lib.effects.pitch_shift(data, sr=44100, n_steps=4)

if __name__ == '__main__':
    app.run(debug=True)