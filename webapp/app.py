from fileinput import filename
import numpy as np
from flask import Flask, render_template,request, url_for, jsonify, redirect
import pickle, os
import speech_recognition as sr
# from keyword_spotting_service import Keyword_Spotting_Service


app = Flask(__name__)
model = pickle.load(open('../model_mb.pkl', 'rb'))

# Landing Page
@app.route("/")
def home():
    return render_template('base.html', transcript="")

# Prediction Page
@app.route('/prediction',methods=["GET", "POST"])
def predict():

    transcript=""

    if request.method == "POST":
        print("FORM DATA RECEIVED")
        
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            recognizer = sr.Recognizer()
            audiofile = sr.AudioFile(file)
            with audiofile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            print(transcript)

    return render_template('base.html', transcript = transcript)


# def predict():
#     #For rendering results on HTML GUI
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0], 2) 
#     return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

