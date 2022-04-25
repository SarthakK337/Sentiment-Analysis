from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
from sentiment_Score import sentimentScore
from preprocess_Text import preprocessText

# Importing Libs
# Data manupulation
import pandas as pd
import numpy as np
import re
import string

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/dataanalysis", methods=['POST'])
@cross_origin()
def dataanalysis():
    """
    here we analyse the data.
    :return: required output
    """
    try:
        if request.json is not None:
            # Load data convert into dataframe
            path = request.json['filepath']
            # df = load_dataset(path)
            df = pd.read_excel(path)

            # Initialization token
            tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
            model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

            # Text cleaning and creating new column
            df['Text_update'] = df["Text"].apply(lambda x: preprocessText(x).preprocess_text())
            df['Text_update'] = df["Text"].apply(lambda x: preprocessText(x).preprocess_text())

            # Filtering Data
            df=df[df["Text_update"]!=""]
            df=df[df.Star<=2].reset_index()
            df.drop(columns="index",inplace=True)

            # Predicting sentiment using pre-trained model
            df['sentiment'] = df['Text_update'].apply(lambda x: sentimentScore(x, tokenizer, model).sentiment_score())

            # Classifying sentiment into 1 and 0
            df["old"] = df.Star.apply(lambda x: 0 if x <= 2 else 1)
            df["new"] = df.sentiment.apply(lambda x: 0 if x <= 2 else 1)
            df["diff"] = df.old - df.new

            # Filter out results
            result = df[df['diff'] != 0]
            result.reset_index(inplace=True)
            result.drop(columns=["index", "sentiment", "old", "new", "diff", "Text_update"], inplace=True)
            result=result.head().to_json(orient="records")

            return Response(json.loads(result))

        elif request.form is not None:
            # Load data convert into dataframe
            path = request.form['filepath']
            # df = load_dataset(path)
            df = pd.read_excel(path)

            # Initialization token
            tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
            model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

            # Text cleaning and creating new column
            df['Text_update'] = df["Text"].apply(lambda x: preprocessText(x).preprocess_text())
            df['Text_update'] = df["Text"].apply(lambda x: preprocessText(x).preprocess_text())

            # Filtering Data
            df=df[df["Text_update"]!=""]
            df=df[df.Star<=2].reset_index()
            df.drop(columns="index",inplace=True)

            # Predicting sentiment using pre-trained model
            df['sentiment'] = df['Text_update'].apply(lambda x: sentimentScore(x, tokenizer, model).sentiment_score())

            # Classifying sentiment into 1 and 0
            df["old"] = df.Star.apply(lambda x: 0 if x <= 2 else 1)
            df["new"] = df.sentiment.apply(lambda x: 0 if x <= 2 else 1)
            df["diff"] = df.old - df.new

            # Filter out results
            result = df[df['diff'] != 0]
            result.reset_index(inplace=True)
            result.drop(columns=["index", "sentiment", "old", "new", "diff", "Text_update"], inplace=True)
            result=result.head().to_json(orient="records")

            return Response(json.loads(result))

        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
