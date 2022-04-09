from os import listdir
from os.path import isfile, join
import re
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import sys, csv

csv.field_size_limit(sys.maxsize)

model = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")


def scorer(model, X, y):
    y_pred = model.predict(X)
    score = f1_score(y_pred, y)
    return score



def tokenize_text(text):
    return ' '.join(tokenizer.tokenize(text))



data = pd.read_csv('data/full_dataframe.csv')
train_labels = np.array(data['label'].values)
data.pop('label')
train_texts = data['text']
data.pop('text')
train_metadata = data.values

train_embeddings = [tokenizer.encode(train_text, padding='max_length', max_length=1024, truncation=True)
               for train_text in train_texts]
train_data = np.concatenate((np.array(train_embeddings), np.array(train_embeddings)), axis=1)
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


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2,
                                                    random_state=13, shuffle=False)

print("Pos Neg report:")
print((np.shape(y_train)[0] - np.sum(y_train)) / np.sum(y_train))
def xgboost_classifier():
    model = xgb.XGBClassifier(scale_pos_weight=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_res(y_pred, y_test, model_name='XGBoost')

xgboost_classifier()
