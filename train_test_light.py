
from __init__ import get_new_experiment_folder, individual_train_path, individual_dev_path, individual_labels_path
from scipy import sparse
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pdb
import pickle
import os

def save_sk_model(model):
    model_name = str(model).split('(')[0]
    experiment_folder = get_new_experiment_folder(model_name)
    model_path = os.path.join(experiment_folder, "best.pkl")
    with open(model_path, 'wb') as fout:
        pickle.dump(model, fout)

def train_individual(mode):
    train_texts_vec = sparse.load_npz(individual_train_path)
    dev_texts_vec = sparse.load_npz(individual_dev_path)
    train_labels, dev_labels = np.load(individual_labels_path, allow_pickle=True)
    
    if mode == "regression":
        train_labels = np.array(train_labels).astype(np.float32)
        dev_labels = np.array(dev_labels).astype(np.float32)

        model = SVR(kernel='linear', verbose=10)

        metric_name = 'MAE'
        metric = mean_absolute_error
    else:
        model = StackingClassifier([
            ("svr", SVC(kernel='linear', verbose=10))
        ], final_estimator=DecisionTreeClassifier(max_depth=1))

        metric_name = 'f1_score'
        metric = lambda *args: f1_score(*args, average='weighted')
    
    model.fit(train_texts_vec, train_labels)
    pred_labels = model.predict(dev_texts_vec)
    metric_result = metric(dev_labels, pred_labels)
    print(f"Model finished with {metric_name}: {metric_result}")

    save_sk_model(model)

def train_time_series():
    # TODO: get_individual_model() from already trained
    # hidden_svm = model.estimators_[0]
    # confidence = hidden_svm.coef_ * x + hidden_svm.intercept_
    pass

if __name__ == "__main__":
    # train_individual(mode="regression")
    train_individual(mode="classification")

