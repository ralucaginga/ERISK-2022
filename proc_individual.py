# process individual and dump it into a list
import json
import re
import pdb
import numpy as np
import os
import pickle

from __init__ import data_json_path, individual_train_path, individual_dev_path, individual_labels_path, vectorizer_path
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")


def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_text(text):
    return ' '.join(tokenizer.tokenize(text))

def get_all_lists(full_data, n_maximum_posts=1800):
    all_posts = []
    all_labels = []
    for i, (_, posts) in enumerate(full_data.items()):
        if i > n_maximum_posts:
            break
        for post in posts:
            if post['text'] != '':
                clean_post = clean_text(post['text'])
                if clean_post != '':
                    tokenized_post = tokenize_text(clean_post)
                    
                    all_posts.append(tokenized_post)
                    all_labels.append(post['label'])
        print(f"Processing progress {i}/{len(full_data.keys())}", end='\r')
    return all_posts, all_labels

def main():
    with open(data_json_path, 'r') as fin:
        full_data = json.load(fin)
    all_posts, all_labels = get_all_lists(full_data)

    train_text, dev_text, train_labels, dev_labels = train_test_split(all_posts, all_labels, random_state=101, test_size=0.1)

    vectorizer = TfidfVectorizer(analyzer='word', lowercase=True, max_features=1000)
    train_texts_vec = vectorizer.fit_transform(train_text)
    dev_texts_vec = vectorizer.transform(dev_text)

    with open(vectorizer_path, 'wb') as fout:
        pickle.dump(vectorizer, fout)

    sparse.save_npz(individual_train_path, train_texts_vec)
    sparse.save_npz(individual_dev_path, dev_texts_vec)

    individual_labels = np.array([train_labels, dev_labels])
    np.save(individual_labels_path, individual_labels)

if __name__ == "__main__":
    main()