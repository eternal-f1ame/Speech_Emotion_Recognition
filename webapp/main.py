from fileinput import filename
import numpy as np
from flask import Flask, render_template,request, url_for, jsonify, redirect
import pickle, os
from copy import deepcopy
import soundfile as sf
import speech_recognition as sr
import librosa
import io
# from keyword_spotting_service import Keyword_Spotting_Service

emotions_decoder = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']


app = Flask(__name__)
model = pickle.load(open('./model_mb.pkl', 'rb'))
mean = None
std = None

def proc_aud(aud, sr, mfcc=False, chroma=True, mel=False):
    X = aud
    sample_rate = sr
    if chroma:
        stft=np.abs(librosa.stft(X))

    result=np.array([])

    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))

    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))

    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    
    if mean is not None and std is not None:
        result = (result - mean) / mean

    return result


# Landing Page
@app.route("/")
def home():
    return render_template('base_new.html', transcript="")

# Prediction Page
@app.route('/prediction',methods=["GET", "POST"])
def predict():

    transcript = ""
    emotion = ""

    if request.method == "POST":
        print("FORM DATA RECEIVED")
        
        if "file" not in request.files:
            print(request.url)
            return redirect('/')
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect('/')

        print(file.filename)

        print(filename)
        
        if file:
            audio = io.BytesIO(file.read())
            audio, samplerate = librosa.load(audio, duration=2.5, offset=0.5)
            audio = proc_aud(audio, samplerate).reshape(1, -1)

            pred = model.predict(audio)

            emotion = emotions_decoder[pred.squeeze()]
            transcript = "sex suxxxx"

            
            '''recognizer = sr.Recognizer()
            audiofile = sr.AudioFile(file_new)
            with audiofile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            print(transcript)'''

    return render_template('base_new.html', transcript = transcript, emotion=emotion)


# def predict():
#     #For rendering results on HTML GUI
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0], 2) 
#     return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run()
