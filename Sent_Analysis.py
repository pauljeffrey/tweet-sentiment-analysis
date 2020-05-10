import gzip
from html import unescape
import time
import pickle

import dill 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import spacy
from spacy.lang.en import STOP_WORDS
nlp = spacy.load('en_core_web_sm',disable = ['ner','parser','tagger'])
words = ['not','so','very','cannot','can']
for word in words:
    STOP_WORDS.remove(word)
    
STOP_WORDS_lemma = [word.lemma_ for word in nlp(' '.join(list(STOP_WORDS)))]
STOP_WORDS_lemma = set(STOP_WORDS_lemma).union({' ,','.',';'})




days = ['monday','tuesday','wednesday', 'thursday', 'friday', 'saturday','sunday','sun','mon','tue','wed','thur','fri','sat']
class OnlinePipeline(Pipeline):
    def partial_fit(self,X,y):
        try:
            Xt = X.copy()
        except AttributeError:
            Xt = X
        
        for _,est in self.steps:
            if hasattr(est,'partial fit') and hasattr(est,'predict'):
                est.partial_fit(Xt,y)

            if hasattr(est,'transform'):
                Xt = est.transform(Xt)

        return self

def fit_model(func):
    def wrapper(*args, **kwargs):
        t_0 = time.time()
        model = func()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_0

        print(f'Training time : {t_elapsed}')
        print(f'Training accuracy : {model.score(X_train,y_train)}')
        print(f'Testing accuracy : {model.score(X_test,y_test)}')

        return model

    return wrapper

def partial_fit(func):
    def wrapper(*args, **kwargs):
        t_0 = time.time()
        model = func()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_0

        print(f'Training time : {t_elapsed}')
        print(f'Training accuracy : {model.score(X_train,y_train)}')
        print(f'Testing accuracy : {model.score(X_test,y_test)}')

        return model

    return wrapper



def remove_day(doc):
    doc = doc.lower()
    for word in doc.split(' '):
        if word in days:
            doc = doc.replace(word,'')
    return doc


def preprocessor(doc):
    doc = remove_day(doc)
    return unescape(doc)

def tokenizer(doc):
    return [word.lemma_ for word in nlp(doc)]


@fit_model
def online_model():
    from sklearn.linear_model import SGDClassifier
    vectorizer = HashingVectorizer(preprocessor=preprocessor,
                                    tokenizer = tokenizer,
                                    alternate_sign = False,
                                    #ngram_range = (1,2),
                                    stop_words = STOP_WORDS_lemma)
    clf = MultinomialNB()
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', clf)])

    return pipe


@fit_model
def online_tfidf_model():
    from sklearn.linear_model import SGDClassifier
    vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                    tokenizer = tokenizer,
                                    ngram_range = (1,2),
                                    stop_words = STOP_WORDS_lemma)
    clf = SGDClassifier(loss='log', max_iter=5)
    pipe = OnlinePipeline([('vectorizer', vectorizer),
                            ('classifier', clf)])

    return pipe


@fit_model
def construct_model():
    vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                tokenizer = tokenizer,
                                #ngram_range = (1,2),
                                stop_words = STOP_WORDS_lemma)

    clf = MultinomialNB()
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', clf)])
    return pipe


def serialize_model(func):
    #model = construct_model()
    model=  func()

    with gzip.open('sentiment_ng_model.dill.gz', 'wb') as f:
        dill.dump(model, f, recurse = True)


def pickle_model(func):
    model = func()
    with open('sentiment_ng_model.pkl','wb') as f:
        pickle.dump(model,f)
    print('Done')

def remove_rare(df):
    print('Removing the rare words....')
    freq = pd.Series(' '.join(df['SentimentText']).split()).value_counts()[-200:]
    df['SentimentText'] = df['SentimentText'].apply(lambda x: ' '.join(x for x in x.split() if x not in freq))
    df['SentimentText'] = df['SentimentText'].apply(lambda x: ' '.join(x for x in x.split() if not x.startswith('@') ))
    
    return df

if __name__ == '__main__':
    print('initiating training......')
    df = pd.read_csv('Sentiment Analysis Dataset.csv', error_bad_lines= False)

    df = remove_rare(df)

    print('Removing punctuations.....')
    X= df['SentimentText'].str.replace('[^\w\s]','')
    y = df['Sentiment']

    splits = train_test_split(X,y,test_size=0.1, random_state=0)
    X_train ,X_test, y_train ,y_test = splits


    #pickle()
    print('Now training and saving the model.....')
    serialize_model(construct_model)
    #pickle_model(construct_model)
    #model = online_model()
    #model = construct_model()
    