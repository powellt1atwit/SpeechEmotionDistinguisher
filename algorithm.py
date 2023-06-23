# %%
import os
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns

import librosa as lib
import librosa.display 

import IPython.display as ipd 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# %%
path = "Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)"
file_list = os.listdir(path)

# %%
#Read files
files = []
modality =[]
vocal =[]
emotion =[]
intensity =[]
statement =[]
repetition =[]
actor =[]
gender = []
time = []
audio_data = []
sr = []

max_row = 0
max_col = 0
min_row = 1000
min_col = 1000

n_fft = 2048
hop_length = 512
n_mels = 200

for file_name in file_list:
    file_path = path+'/'+file_name
    mod, voc, emo, inten, state, repe, act = file_name.split('-')
    act = act.split('.')[0]
    
    if emo != '02':
        #store metadata
        files.append(file_name)
        modality.append(mod)
        vocal.append(voc)
        intensity.append(inten)
        statement.append(state)
        repetition.append(repe)
        actor.append(act)

        if (emo == '01'):
            emotion.append('neutral')
        elif (emo == '03'):
            emotion.append('happy')
        elif (emo == '04'):
            emotion.append('sad')
        elif (emo == '05'):
            emotion.append('angry')
        elif (emo == '06'):
            emotion.append('fearful')
        elif (emo == '07'):
            emotion.append('disgust')
        elif (emo == '08'):
            emotion.append('surprised')

        if (int(act) % 2 == 0):
            gender.append(1) #female
        else:
            gender.append(2) #male

        audio, sfreq = lib.load(file_path, sr=44100,offset=0.5,duration = 3.5)
        time_line = np.arange(0,len(audio)) / sfreq
        time.append(time_line)
        audio_data.append(audio)
        sr.append(sfreq)

        mfccs = librosa.feature.mfcc(y=audio, sr=sfreq, n_mfcc=13)
        mel_spec = lib.feature.melspectrogram( y = audio, sr=sfreq, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = lib.power_to_db(mel_spec, ref=np.max)

        if (S_DB.shape[0] > max_row):
            max_row = S_DB.shape[0]

        if (S_DB.shape[1] > max_col):
            max_col = S_DB.shape[1]

        if (S_DB.shape[0] < min_row):
            min_row = S_DB.shape[0]

        if (S_DB.shape[1] < min_col):
            min_col = S_DB.shape[1]

# %%
mfcc_np = np.zeros((len(files), max_row, max_col))
df_files = pd.DataFrame({'file_name': files, 'emotion': emotion, 'intensity': intensity, 
                         'actor': actor, 'gender': gender, 'sfreq': sr, 'audio': audio_data })
df_files.emotion.unique()

# %%
#Extract audio data and labels
audio_list = []

for file_name in file_list:
    file_path = path+'/'+file_name
    mod, voc, emo, inten, state, repe, act = file_name.split('-')
    act = act.split('.')[0]
    
    if emo != '02':     
        X, sample_rate = lib.load(file_path,sr=44100,offset=0.5,duration = 3.5)
        audio_list.append(X)

only_audio_array = np.asarray(audio_list)
df_label = df_files[['emotion']].copy()
le = preprocessing.LabelEncoder()
df_label.emotion = le.fit_transform(df_label.emotion)

# %%
#Split between train and test and ENCODE
random_num = 7 #random.randint(0,100) 
print("Random Number is ", random_num)

X_train, X_test, y_train, y_test = train_test_split((only_audio_array)
                                                    , df_label
                                                    , test_size=0.2
                                                    , shuffle=True
                                                    , random_state=random_num)
enc = OneHotEncoder()
label_onehot_train = enc.fit_transform(y_train).toarray()
label_onehot_test = enc.fit_transform(y_test).toarray()
print(enc.categories_)

#%%
#Augment data with noise
def manipulate_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

a_list = [1, 2]
distribution = [0.0, 1.0]

manipulated_audio = []
manipulated_onehot_label = []

for i in range(len(X_train)):
    random_number = np.random.choice(a_list, p = distribution)
    if random_number == 2:
        data = manipulate_noise(X_train[i],0.008)
        manipulated_audio.append(data)
        manipulated_onehot_label.append(label_onehot_train[i])
        
a_list = [1, 2]
distribution = [0.0, 1.0]

# %%
#Augment data with pitch
def manipulate_pitch(data):
    return lib.effects.pitch_shift(data, sr=44100, n_steps=4)
    
a_list = [1, 2]
distribution = [0.1, 0.9]

manipulated_audio_pitch = []
manipulated_onehot_label_pitch = []

for i in range(len(X_train)):
    random_number = np.random.choice(a_list, p = distribution)
    if random_number == 2:
        data = manipulate_pitch(X_train[i])
        manipulated_audio_pitch.append(data)
        manipulated_onehot_label_pitch.append(label_onehot_train[i])

# %%
X_train_manu = np.concatenate((X_train,np.asarray(manipulated_audio),
                               np.asarray(manipulated_audio_pitch)),axis = 0)


label_onehot_train_manu = np.concatenate((label_onehot_train,np.asarray(manipulated_onehot_label),
                                          np.asarray(manipulated_onehot_label_pitch)),axis = 0)

# %%
#Change audio data to spectrograms
spectogram_list_train = []
for item in X_train_manu:
    mel_spec = lib.feature.melspectrogram(y = item, sr=44100, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = lib.power_to_db(mel_spec, ref=np.max)

    if S_DB.shape[1] < max_col:
        result = np.zeros((max_row,max_col - S_DB.shape[1]),dtype=float)
        a = np.hstack((S_DB,result))
        spectogram_list_train.append(a)
    else:
        spectogram_list_train.append(S_DB)
        
spectogram_array_train = np.asarray(spectogram_list_train)

spectogram_list_test = []
for item in X_test:
    mel_spec = lib.feature.melspectrogram(y = item, sr=44100, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = lib.power_to_db(mel_spec, ref=np.max)

    if S_DB.shape[1] < max_col:
        result = np.zeros((max_row,max_col - S_DB.shape[1]),dtype=float)
        a = np.hstack((S_DB,result))
        spectogram_list_test.append(a)
    else:
        spectogram_list_test.append(S_DB)
        
spectogram_array_test = np.asarray(spectogram_list_test)

# %%
#Data normalization
mean = np.mean(spectogram_array_train,axis = 0)
std = np.std(spectogram_array_train,axis = 0)

x_train_temp = spectogram_array_train.reshape(spectogram_array_train.shape[0],spectogram_array_train.shape[1]*spectogram_array_train.shape[2])
x_test_temp = spectogram_array_test.reshape(spectogram_array_test.shape[0],spectogram_array_test.shape[1]*spectogram_array_test.shape[2])

scaler = preprocessing.StandardScaler().fit(x_train_temp)

X_train_norm = scaler.transform(x_train_temp) 
X_test_norm = scaler.transform(x_test_temp)

X_train_norm_re = X_train_norm.reshape(X_train_norm.shape[0],spectogram_array_train.shape[1],spectogram_array_train.shape[2])
X_test_norm_re = X_test_norm.reshape(X_test_norm.shape[0],spectogram_array_test.shape[1],spectogram_array_test.shape[2])

X_test_norm_re_n, X_vald, label_onehot_test_n, y_vald = train_test_split((X_test_norm_re)
                                                    , label_onehot_test
                                                    , test_size=0.1
                                                    , shuffle=True
                                                    , random_state=random_num)

#%%
#Convert to array
y_train = np.array(y_train)
y_test = np.array(y_test)

# %%
#Construct CNN-LSTM layers
num_of_emotions = len(le.classes_)

model = keras.Sequential()
model.add(keras.Input(shape=(X_train_norm_re.shape[1],X_train_norm_re.shape[2])))

model.add(layers.Conv1D(64,3, activation="relu",padding="same",strides = 1))

model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.elu))
model.add(layers.MaxPooling1D(pool_size = 4))

model.add(layers.Conv1D(128,3, activation="relu",padding="same",strides = 1))

model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.elu))
model.add(layers.MaxPooling1D(pool_size = 4))

model.add(layers.Conv1D(256,3, activation="relu",padding="same",strides = 1))

model.add(layers.BatchNormalization())
model.add(layers.Activation(activations.elu))
model.add(layers.MaxPooling1D(pool_size = 4))

model.add(layers.LSTM(540,return_sequences=True))
model.add(layers.LSTM(256,return_sequences=True))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(50, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dense(20, activation="relu"))
model.add(layers.Dense(num_of_emotions, activation="softmax"))

model.summary()

# %%
#Implement early stopping and run CNN-LSTM
es = EarlyStopping(monitor='val_accuracy', patience=200,restore_best_weights=True,verbose=1)
filepath_to_save = "model.hdf5"
checkpoint = ModelCheckpoint(filepath_to_save, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]

model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

hist = model.fit(X_train_norm_re, label_onehot_train_manu, batch_size=200
                           ,epochs=150, validation_data=(X_test_norm_re, label_onehot_test),callbacks=callbacks, shuffle="true")

# %%
#Plot loss of model
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
#Plot accuracy of model
plt.clf()
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# %%
# Evaluate model on private test set
score = model.evaluate(X_test_norm_re, label_onehot_test, verbose=0)
print ("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# %%
#Add weights to CNN-LSTM model and get accuracy
filepath_to_save = "model.hdf5"

model.load_weights(filepath_to_save)
model.compile(loss="categorical_crossentropy", optimizer="adamax", metrics=["accuracy"])

score = model.evaluate(X_test_norm_re, label_onehot_test, verbose=0)
print ("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# %%
def maxposition(A):
    A = list(A)
    maxposition = A.index(max(A)) 
    return maxposition

# %%
#Get accuracy of max position list
y_pred = model.predict(X_test_norm_re)

y_pred_adj = []
for row in y_pred:
    y_pred_adj.append(maxposition(row))
    

y_test_encode = enc.inverse_transform(label_onehot_test)
print("accuracy: %s" % (accuracy_score(y_test_encode, y_pred_adj)))

# %%
#Construct confusion matrix
y_true = y_test_encode #label_encoder.inverse_transform(y_test_encode.ravel())
y_pred = y_pred_adj #label_encoder.inverse_transform(y_pred)
y_pred

data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.axes_style("whitegrid")
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

# %%
#Reshape data for SVM
X_train_norm_reshape_sk = X_train_norm_re.reshape(X_train_norm_re.shape[0],
                                                  X_train_norm_re.shape[1]*X_train_norm_re.shape[2])
X_test_norm_reshape_sk = X_test_norm_re.reshape(X_test_norm_re.shape[0],
                                                  X_test_norm_re.shape[1]*X_test_norm_re.shape[2])


y_train_encode_sk = enc.inverse_transform(label_onehot_train_manu)
y_test_encode_sk = enc.inverse_transform(label_onehot_test)

y_train_encode_sk = y_train_encode_sk.ravel()
y_test_encode_sk = y_test_encode_sk.ravel()

print(X_train_norm_reshape_sk.shape)
print(X_test_norm_reshape_sk.shape)
print(y_train_encode_sk.shape)
print(y_test_encode_sk.shape)

# %%
#Run RBF SVM
clf = SVC(C = 5, kernel = 'rbf', gamma = "auto", probability=True, random_state=random_num)
clf.fit(X_train_norm_reshape_sk, y_train_encode_sk)
result_svm = clf.predict(X_test_norm_reshape_sk)

print("accuracy = %s" % (accuracy_score(y_test_encode_sk, result_svm)))

# %%
#Construct confusion matrix
y_true = y_test_encode_sk
y_pred = result_svm
y_pred


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.axes_style("whitegrid")
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

#%%
#Run poly SVM
poly = SVC(C = 5, kernel = 'poly', gamma = "auto",  probability=True, random_state=random_num, degree=5)
poly.fit(X_train_norm_reshape_sk, y_train_encode_sk)
result_poly = poly.predict(X_test_norm_reshape_sk)

print("accuracy = %s" % (accuracy_score(y_test_encode_sk, result_poly)))

# %%
#Construct confusion matrix
y_true = y_test_encode_sk
y_pred = result_poly
y_pred


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.axes_style("whitegrid")
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

# %%
#Run sigmoid SVM
sig = SVC(C = 5, kernel = 'sigmoid', gamma = "auto",  probability=True, random_state=random_num)
sig.fit(X_train_norm_reshape_sk, y_train_encode_sk)
result_sig = sig.predict(X_test_norm_reshape_sk)

print("accuracy = %s" % (accuracy_score(y_test_encode_sk, result_sig)))

#%%
#Construct confusion matrix
y_true = y_test_encode_sk
y_pred = result_sig
y_pred


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.axes_style("whitegrid")
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

# %%
#Save and Load file from directory
import pickle

svm_path = "svm_model.pkl"
with open(svm_path, 'wb') as file:
    pickle.dump(clf, file)

svm_path = "svm_model.pkl"
with open(svm_path, 'rb') as file:
    pickle_model = pickle.load(file)

Ypredict = pickle_model.predict(X_test_norm_reshape_sk)

# %%
#Run Bagging SVM
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(estimator=SVC(C = 5, kernel = 'rbf', gamma = "auto", probability=True, random_state=random_num), n_estimators=10, random_state=random_num)
bag.fit(X_train_norm_reshape_sk, y_train_encode_sk)
result_bag = bag.predict(X_test_norm_reshape_sk)

print("accuracy = %s" % (accuracy_score(y_test_encode_sk, result_bag)))

#%%
#Construct confusion matrix
y_true = y_test_encode_sk
y_pred = result_bag
y_pred


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.axes_style("whitegrid")
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

#%%