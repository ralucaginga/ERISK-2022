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
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import torch

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
    false_positives = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    print(f"""
Model: {model_name}
Accuracy score: {accuracy_score(y_pred, y_test)}
Precision score: {precision_score(y_pred, y_test)}
Recall score: {recall_score(y_pred, y_test)}
F1 score: {f1_score(y_pred, y_test)}
False positives: {false_positives}
False negatives: {false_negatives}

""")



# Shuffling for cross_val_score
np.random.seed(3)
idx = np.random.permutation(len(train_data))
train_data = train_data[idx]
train_labels = train_labels[idx]



X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2,
                                                    random_state=3, shuffle=True)

print("Pos Neg report:")
print((np.shape(y_train)[0] - np.sum(y_train)) / np.sum(y_train))

def create_avg_data():
    data_copy = data.copy()
    columns = 'adjective_count,verb_count,noun_count,adverb_count,negation_count,first_person_pron_count,antidepress_count,three_grams_count,five_grams_count,overgeneralization_count,psychoactive_count,unpleasant_feel_count,nssi_count,temporal_count,punct_count,questions_count,exclamations_count,capitalized_count'.split(',')
    for column in columns:
        data_copy[column] = data_copy[column] / data_copy['words_count']
    data_copy.pop('words_count')
    print(data_copy.keys())
    return np.array(data_copy.values)[idx]


def evaluate_models():
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    scores = cross_val_score(xgb_model, train_data, train_labels,
                             cv=5, scoring=scorer)
    print('XGBoost cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))

    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    train_data_avg = create_avg_data()
    scores = cross_val_score(xgb_model, train_data_avg, train_labels,
                             cv=5, scoring=scorer)
    print('XGBoost 2 cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))

    lgbm_model = LGBMClassifier()
    scores = cross_val_score(lgbm_model, train_data, train_labels,
                             cv=5, scoring=scorer)
    print('LightGBM cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))

    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    scores = cross_val_score(svm_model, train_data, train_labels,
                             cv=5, scoring=scorer)
    print('SVM cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))
    logistic_regression = LogisticRegression(max_iter=100000)
    scores = cross_val_score(logistic_regression, train_data, train_labels,
                             cv=5, scoring=scorer)
    print('Logistic Regression cross validation:')
    print(scores)
    print(np.mean(np.array(scores)))

def save_models():
    xgb_model = xgb.XGBClassifier(scale_pos_weight=10)
    xgb_model.fit(train_data, train_labels)
    pickle.dump(xgb_model, open('bogdan_pickle_xgb_on_metadata.sav', 'wb'))

    xgb_model_avg = xgb.XGBClassifier(scale_pos_weight=10)
    train_data_avg = create_avg_data()
    print(np.shape(train_data_avg))
    xgb_model_avg.fit(train_data_avg, train_labels)
    pickle.dump(xgb_model_avg, open('bogdan_pickle_xgb_on_metadata_avg.sav', 'wb'))

    lgbm_model = LGBMClassifier()
    lgbm_model.fit(train_data, train_labels)
    pickle.dump(lgbm_model, open('bogdan_pickle_lgbm_on_metadata.sav', 'wb'))
    # torch.save(lgbm_model.state_dict(), 'bogdan_pytorch_lgbm_on_metadata.pth')


# evaluate_models()
save_models()