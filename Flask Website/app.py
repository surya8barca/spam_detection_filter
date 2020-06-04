from flask import Flask,jsonify,request
import joblib
import pandas as pd
from flask.templating import render_template
import string
from nltk.corpus import stopwords

app = Flask(__name__)


def clean_text(str):
    
    clean_pun=[ch for ch in str if ch not in string.punctuation]
    clean_pun1=''.join(clean_pun)
        
    clean_stop1=list(clean_pun1.split( ))
    clean_str=[x for x in clean_stop1 if x not in stopwords.words('english')]
        
    return clean_str

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict/",methods=['GET'])
def predict():
    result=request.args
    model=joblib.load('model.sav')
    prediction=model.predict([result["message"]])
    return jsonify({'result': prediction[0]})


if __name__ == '__main__':
    app.run()