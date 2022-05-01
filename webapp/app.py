import numpy as np
from flask import Flask, render_template,request, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('../model_mb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('base.html')

# @app.route('/',methods=['POST'])

# def predict():
#     #For rendering results on HTML GUI
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0], 2) 
#     return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

