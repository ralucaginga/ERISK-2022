# F1 score 0.68

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request

bert_tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")



def scorer(model, X, y):
    y_pred = model.predict(X)
    score = f1_score(y_pred, y)
    return score


data = pd.read_csv('data/full_dataframe.csv')
labels = np.array(data['label'].values)
data.pop('label')
texts = data['text']
data.pop('text')
metadata = data.values
data = np.array(metadata)


def print_res(y_test, y_pred, model_name=""):
    print(f"""
Model: {model_name}
Accuracy score: {accuracy_score(y_pred, y_test)}
Precision score: {precision_score(y_pred, y_test)}
Recall score: {recall_score(y_pred, y_test)}
F1 score: {f1_score(y_pred, y_test)}

""")


# Shuffling
np.random.seed(3)
idx = np.random.permutation(len(data))
data = data[idx]
labels = labels[idx]



def xgboost_classifier():
    global data, labels, texts
    idx_train, idx_test, _, _ = train_test_split([i for i in range(0, len(data))], data, test_size=0.2,
                                                 random_state=13, shuffle=True)
    X_train = data[idx_train]
    X_test = data[idx_test]
    y_train = labels[idx_train]
    y_test = labels[idx_test]
    train_texts = texts[idx_train]
    test_texts = texts[idx_test]

    print("Pos Neg report:")
    print((np.shape(y_train)[0] - np.sum(y_train)) / np.sum(y_train))
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_emotion = []
    y_pred_sentiment = []
    for task in []:
        model_path = f"cardiffnlp/twitter-roberta-base-{task}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)


        tokenizer = AutoTokenizer.from_pretrained(model_path)
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        print('Test texts len')
        print(len(test_texts))
        for (text_idx, text) in enumerate(test_texts):
            text = text[60:]
            if text_idx % 100 == 0:
                print(f'Got to {text_idx}')
            encoded_input = tokenizer(text, return_tensors='pt')
            #try:
            output = model(**encoded_input)
            scores = softmax(output[0][0].detach().numpy())
            ranking = np.argsort(scores)[::-1]
            top_label = labels[ranking[0]]
            likely_label = 0

            if top_label in ['sadness', 'negative']:
                likely_label = 1
            if task == 'emotion':
                y_pred_emotion.append(likely_label)
            if task == 'sentiment':
                y_pred_sentiment.append(likely_label)
    y_pred_emotion = np.array(y_pred_emotion)
    y_pred_sentiment = np.array(y_pred_sentiment)
    y_composed = []
    for y_idx in range(0, y_pred_sentiment.size):
        label = y_pred_xgb[y_idx]
        if y_pred_sentiment[y_idx] == 1 and y_pred_emotion[y_idx] == 1:
            label = 1
        y_composed.append(label)
    y_composed = np.array(y_composed)
    print_res(y_pred_xgb, y_test, model_name='XGBoost')
    print_res(y_pred_emotion, y_test, model_name='ROBERTA base emotion')
    print_res(y_pred_sentiment, y_test, model_name='ROBERTA base sentiment')
    print_res(y_composed, y_test, model_name='Composed models')



def run_classifiers():
    full_train = data
    for task in ['sentiment', 'emotion']:
        model_path = f"cardiffnlp/twitter-roberta-base-{task}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(f'Current task is {task}')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_scores_train = []
        for (text_idx, text) in enumerate(texts):
            text = ' '.join(tokenizer.tokenize(text)[:60])
            if text_idx % 100 == 0:
                print(f'Got to {text_idx}')
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = softmax(output[0][0].detach().numpy())
            model_scores_train.append(scores)
        model_scores_train = np.array(model_scores_train)
        full_train = np.concatenate((full_train, model_scores_train), axis=1)

    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    scores = cross_val_score(svm_model, full_train, labels, cv=5, scoring=scorer)
    print('SVM cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    scores = cross_val_score(xgb_model, full_train, labels, cv=5, scoring=scorer)
    print('XGBoost cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))
    lgbm_model = LGBMClassifier()
    scores = cross_val_score(lgbm_model, full_train, labels, cv=5, scoring=scorer)
    print('LightGBM cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))
    logistic_regression = LogisticRegression(max_iter=100000)
    scores = cross_val_score(logistic_regression, full_train, labels, cv=5, scoring=scorer)
    print('Logistic Regression cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))

run_classifiers()
