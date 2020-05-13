from flask import Flask, request ,redirect, render_template,url_for

import gzip
from os.path import join,abspath,curdir
from pathlib import Path
import dill

app = Flask(__name__)

'''This webapp basically works by loading trained model from dill file when it receives a post request
    and predicting the sentiment of the text passed with the post request.
    App also contains links to my completed machine learning projects.'''

#Create platform specific absolute path to serialized model
cwd = abspath(curdir)
model_path = join(cwd, 'sentiment_ng_model.dill.gz')
path = Path(model_path)


@app.route('/about')
def about():
    return render_template('developer.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')


@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

#This is the function that handles prediction of tweets passed in the post request.
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        tweet = request.args.get('tweet')
    else:
        tweet = request.form['text']
    
    #Load trained model from file
    with gzip.open(path, 'rb') as f:
        model = dill.load(f)

    #predict and round up the probability result of the prediction.
    proba = round(model.predict_proba([tweet])[0,1]* 100,2)
    if (proba <= 52 and proba >= 48):
        sent = -1
    elif proba > (100 - proba):
        sent = 1
    else: 
        sent = 0

    #return results
    return render_template('predict.html', sent = sent , proba= proba)



if __name__ == '__main__':
    app.run()