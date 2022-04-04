from __init__ import data_json_path, vectorizer_path, logs_folder, time_series_data_path
from proc_individual import clean_text, tokenize_text
from sklearn.model_selection import train_test_split

import pdb
import json
import pickle
import numpy as np
import os

def get_individual_model(experiment_path):
    model_path = os.path.join(experiment_path, 'best.pkl')
    with open(model_path, 'rb') as fin:
        hidden_model = pickle.load(fin)
    return hidden_model

def get_all_lists_as_sequence(full_data, window_size=16, window_gap=8):
    all_posts = []
    all_labels = []
    for i, (_, posts) in enumerate(full_data.items()):
        
        this_posts = []
        for post in posts:
            if post['text'] != '':
                clean_post = clean_text(post['text'])
                tokenized_post = tokenize_text(clean_post)
                
                this_posts.append(tokenized_post)
                this_label = post['label']
        
        for window_start in range(0, len(this_posts) - window_size, window_gap):
            all_posts.append(this_posts[window_start: window_start + window_size])
            all_labels.append(this_label)
        
        print(f"Processing time series progress {i}/{len(full_data.keys())}", end='\r')
    print()
    return all_posts, all_labels

def main():
    with open(data_json_path, 'r') as fin:
        full_data = json.load(fin)
    all_posts, all_labels = get_all_lists_as_sequence(full_data)

    with open(vectorizer_path, 'rb') as fin:
        vectorizer = pickle.load(fin)
    experiment_path = os.path.join(logs_folder, 'SVC_0')
    hidden_model = get_individual_model(experiment_path)
    hidden_model_predict = lambda x: x.toarray() @ hidden_model.coef_.T + hidden_model.intercept_

    dataset_confidence = []
    for i, post in enumerate(all_posts):
        posts_vec = vectorizer.transform(post)
        confidence = hidden_model_predict(posts_vec)
        dataset_confidence.append(confidence[:, 0])
        print(f"Vectorizing progress {i}/{len(all_posts)}", end='\r')
    print()

    train_text, dev_text, train_labels, dev_labels = train_test_split(np.array(dataset_confidence), np.array(all_labels), random_state=101, test_size=0.1)
    time_series_dataset = np.array([train_text, dev_text, train_labels, dev_labels], dtype=object)
    np.save(time_series_data_path, time_series_dataset)

    # TODO: make another script for plotting the time series data

if __name__ == "__main__":
    main()