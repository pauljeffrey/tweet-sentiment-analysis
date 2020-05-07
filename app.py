from flask import Flask, request ,redirect, render_template,url_for

import gzip
import dill

app = Flask(__name__)

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


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        tweet = request.args.get('tweet')
    else:
        tweet = request.form['text']
    with gzip.open('sentiment_hash_model.dill.gz', 'rb') as f:
        model = dill.load(f)
    proba = round(model.predict_proba([tweet])[0,1]* 100,2)
    if (proba <= 52 and proba >= 48):
        sent = -1
    elif proba > (100 - proba):
        sent = 1
    else: 
        sent = 0
    
    return render_template('predict.html', sent = sent , proba= proba)
 



if __name__ == '__main__':
    app.run()