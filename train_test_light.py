
from __init__ import get_new_experiment_folder, individual_train_path, individual_dev_path, individual_labels_path, time_series_data_path, logs_folder, vectorizer_path, pictures_folder
from scipy import sparse
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
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

def train_test_a_model(mode, train_texts_vec, train_labels, dev_texts_vec, dev_labels):
    if mode == "regression":
        train_labels = np.array(train_labels).astype(np.float32)
        dev_labels = np.array(dev_labels).astype(np.float32)

        model = SVR(kernel='linear', verbose=10)

        metric_name = 'MAE'
        metric = mean_absolute_error
    else:
        model = SVC(kernel='linear', verbose=10)

        metric_name = 'f1_score'
        metric = lambda *args: f1_score(*args, average='weighted')
    
    model.fit(train_texts_vec, train_labels)
    pred_labels = model.predict(dev_texts_vec)
    metric_result = metric(dev_labels, pred_labels)
    print(f"Model finished with {metric_name}: {metric_result}")

    save_sk_model(model)

def train_individual(mode):
    train_texts_vec = sparse.load_npz(individual_train_path)
    dev_texts_vec = sparse.load_npz(individual_dev_path)
    train_labels, dev_labels = np.load(individual_labels_path, allow_pickle=True)

    train_test_a_model(mode, train_texts_vec, train_labels, dev_texts_vec, dev_labels)

def train_time_series(mode):
    train_text, dev_text, train_labels, dev_labels = np.load(time_series_data_path, allow_pickle=True)
    train_test_a_model(mode, train_text, train_labels, dev_text, dev_labels)

def plot_binary_classification_importance(label, word_names, features_importance, indices_):
    plt.figure(figsize=(20 * len(indices_) / 10, 7))
    plt.bar(word_names[indices_], features_importance[indices_])
    plt.title(f"{label.title()} Orientation")
    
    figure_path = os.path.join(pictures_folder, f"{label}_keywords.png")
    plt.savefig(figure_path)

def plot_svm_binary_classification_viz(vectorizer, model, topk=50):
    word_names = vectorizer.get_feature_names()
    word_names = np.array(word_names)
    
    features_importance = model.coef_.toarray()[0]
    sorted_indices = np.argsort(features_importance)

    worst_indices = sorted_indices[:topk]
    best_indices = sorted_indices[-topk:]

    plot_binary_classification_importance('negative', word_names, features_importance, worst_indices)
    plot_binary_classification_importance('positive', word_names, features_importance, best_indices)

def plot_feature_importance():
    exp_dir = os.path.join(logs_folder, 'SVC_0')
    model_path = os.path.join(exp_dir, 'best.pkl')
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    with open(vectorizer_path, 'rb') as fin:
        vectorizer = pickle.load(fin)
    
    plot_svm_binary_classification_viz(vectorizer, model)

if __name__ == "__main__":
    # train_individual(mode="regression")
    # train_individual(mode="classification")
    # train_time_series(mode="classification")
    plot_feature_importance()

