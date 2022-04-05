import json
import re
import numpy as np
import csv
import xgboost as xgb

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import sparse
import xml.etree.ElementTree as ET

model = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text


def tokenize_text(text):
    return ' '.join(tokenizer.tokenize(text))


train_ids = []
train_data = []
train_labels = []
with open('data/risk_golden_truth.txt', 'r') as file:
    in_file = csv.reader(file, delimiter=' ')
    for line in in_file:
        train_ids.append(line[0])
        train_labels.append(int(line[1]))

sum = 0
for label in train_labels:
    sum += label
print(sum)
print(len(train_labels))
print(sum/len(train_labels))

train_ids = train_ids[:400]
train_data = train_data[:400]
train_labels = train_labels[:400]

for train_id in train_ids:
    root = ET.parse(f'data/{train_id}.xml').getroot()
    writings = root[1:]
    texts = []
    for writing in writings:
        if isinstance(writing[2].text, str):
          texts.append(clean_text(writing[2].text))
    train_data.append(tokenizer.encode(' '.join(texts), padding='max_length', max_length=1024, truncation=True))
print(train_data)
train_data = np.array(train_data)
train_labels = np.array(train_labels)


def print_res(y_test, y_pred, model_name=""):
    print(f"""
Model: {model_name}
Accuracy score: {accuracy_score(y_pred, y_test)}
Precision score: {precision_score(y_pred, y_test)}
Recall score: {recall_score(y_pred, y_test)}
F1 score: {f1_score(y_pred, y_test)}

""")


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=5)



def xgboost_classifier():
    model = xgb.XGBClassifier()
    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_test))
    print(np.shape(y_test))
    for sentence in X_train[1:]:
        if np.shape(sentence) != np.shape(X_train[0]):
            print('Something is wrong')
            print(np.shape(sentence))
            print(np.shape(X_train[0]))
    model.fit(X_train, y_train, )
    y_pred = model.predict(X_test)
    print_res(y_pred, y_test, model_name='XGBoost')

xgboost_classifier()
