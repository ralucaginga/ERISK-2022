# F1 score 0.68

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request


def scorer(model, X, y):
    y_pred = model.predict(X)
    score = f1_score(y_pred, y)
    return score


data = pd.read_csv('data/full_dataframe.csv')
train_labels = np.array(data['label'].values)
data.pop('label')
train_texts = data['text']
data.pop('text')
train_metadata = data.values
train_data = np.array(train_metadata)


def print_res(y_test, y_pred, model_name=""):
    print(f"""
Model: {model_name}
Accuracy score: {accuracy_score(y_pred, y_test)}
Precision score: {precision_score(y_pred, y_test)}
Recall score: {recall_score(y_pred, y_test)}
F1 score: {f1_score(y_pred, y_test)}

""")



# Shuffling for cross_val_score
model = xgb.XGBClassifier(scale_pos_weight=10)
idx = np.random.permutation(len(train_data))
train_data = train_data[idx]
train_labels = train_labels[idx]

scores = cross_val_score(model, train_data, train_labels,
                         cv=5, scoring=scorer)

print('Sklearn cross validation:')
print(scores)
print(np.mean(np.array(scores)))


idx_train, idx_test, _, _ = train_test_split([i for i in range(0, len(train_data))], train_data, test_size=0.2,
                                                    random_state=13, shuffle=True)

X_train = train_data[idx_train]
X_test = train_data[idx_test]
y_train = train_labels[idx_train]
y_test = train_labels[idx_test]
test_texts = train_texts[idx_test]


print("Pos Neg report:")
print((np.shape(y_train)[0] - np.sum(y_train)) / np.sum(y_train))


def xgboost_classifier():
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_emotion = []
    y_pred_sentiment = []
    max_len = 514
    for task in ['sentiment', 'emotion']:
        model_path = f"cardiffnlp/twitter-roberta-base-{task}"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)


        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print('Test texts len')
        print(len(test_texts))
        for (text_idx, text) in enumerate(test_texts):
            text = text[:300]
            if text_idx % 100 == 0:
                print(f'Got to {text_idx}')
            encoded_input = tokenizer(text, return_tensors='pt')
            mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
            with urllib.request.urlopen(mapping_link) as f:
                html = f.read().decode('utf-8').split("\n")
                csvreader = csv.reader(html, delimiter='\t')
            labels = [row[1] for row in csvreader if len(row) > 1]
            #try:
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            top_label = labels[ranking[0]]
            likely_label = 0

            if top_label in ['sadness', 'negative']:
                likely_label = 1
            if task == 'emotion':
                y_pred_emotion.append(likely_label)
            if task == 'sentiment':
                y_pred_sentiment.append(likely_label)
            """except:
                if task == 'emotion':
                    y_pred_emotion.append(0)
                if task == 'sentiment':
                    y_pred_sentiment.append(0)"""
    y_pred_emotion = np.array(y_pred_emotion)
    y_pred_sentiment = np.array(y_pred_sentiment)
    y_composed = []
    for y_idx in range(0, y_pred_sentiment.size):
        sum = y_pred_xgb[y_idx] + y_pred_sentiment[y_idx] + y_pred_emotion[y_idx]
        if sum < 2:
            y_composed.append(0)
        else:
            y_composed.append(1)
    y_composed = np.array(y_composed)
    print_res(y_pred_xgb, y_test, model_name='XGBoost')
    print_res(y_pred_emotion, y_test, model_name='ROBERTA base emotion')
    print_res(y_pred_sentiment, y_test, model_name='ROBERTA base sentiment')
    print_res(y_composed, y_test, model_name='Composed models')


def svm_classifier():
    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print_res(y_pred, y_test, model_name='SVM')


def random_forest_classifier():
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print_res(y_pred, y_test, model_name='Random forest')

xgboost_classifier()
# svm_classifier()
# random_forest_classifier()
