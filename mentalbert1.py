import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import sys, csv

csv.field_size_limit(sys.maxsize)

model = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")


data = pd.read_csv('data/full_dataframe.csv')
train_labels = np.array(data['label'].values)
data.pop('label')
train_texts = data['text']
data.pop('text')
train_metadata = data.values


def process_with_bert(train_texts, train_labels, train_metadata):
    max_len = 512

    new_encodings = []
    new_labels = []
    new_metadata = []
    for idx, (text, label) in enumerate(zip(train_texts, train_labels)):
        all_tokens = tokenizer.tokenize(text)
        n_tokens = len(all_tokens)

        for i in range(0, n_tokens, max_len):
            current_tokens = all_tokens[i:i + max_len]
            current_text = ' '.join(current_tokens)
            current_text = current_text.replace(' ##', '')
            current_encoding = tokenizer.encode(current_text, padding='max_length', max_length=max_len, truncation=True)

            new_encodings.append(current_encoding)
            new_labels.append(label)
            new_metadata.append(train_metadata[idx])
    return np.array(new_encodings), np.array(new_labels), np.array(new_metadata)


def scorer(model, X, y):
    y_pred = model.predict(X)
    score = f1_score(y_pred, y)
    return score



def tokenize_text(text):
    return ' '.join(tokenizer.tokenize(text))





def print_res(y_test, y_pred, model_name=""):
    print(f"""
Model: {model_name}
Accuracy score: {accuracy_score(y_pred, y_test)}
Precision score: {precision_score(y_pred, y_test)}
Recall score: {recall_score(y_pred, y_test)}
F1 score: {f1_score(y_pred, y_test)}

""")


idxs = range(0, len(train_texts))
train_idxs, test_idxs, y_train, y_test = train_test_split(idxs, train_labels, test_size=0.2,
                                                    random_state=13, shuffle=True)

print("Pos Neg report:")
print((np.shape(y_train)[0] - np.sum(y_train)) / np.sum(y_train))
def xgboost_classifier():
    model = xgb.XGBClassifier(scale_pos_weight=10)
    X_train = train_texts[train_idxs]
    y_train = train_labels[train_idxs]
    meta_train = train_metadata[train_idxs]
    (X_train, y_train, meta_train) = process_with_bert(X_train, y_train, meta_train)
    full_train = np.concatenate((X_train, meta_train), axis=1)
    model.fit(full_train, y_train)
    y_pred = []
    X_test = train_texts[test_idxs]
    meta_test = train_metadata[test_idxs]
    for (text, meta) in zip(X_test, meta_test):
        (embeddings, labels, metas) = process_with_bert(np.array([text]), np.array([0]), np.array([meta]))
        current_train = np.concatenate((embeddings, metas), axis=1)
        current_ys = model.predict(current_train)
        likely_label = 0
        for current_y in current_ys:
            if current_y == 1:
                likely_label = 1
        y_pred.append(likely_label)

    print_res(y_pred, y_test, model_name='XGBoost')

xgboost_classifier()

