import json
import pickle
import pdb
import time
import os

import numpy as np
import requests
import re
import spacy
import pandas as pd
import contractions
import numpy as np
import requests
import torch
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from datetime import datetime

from transformers import BertTokenizer, BertForSequenceClassification

from feature_pipeline import features_pipeline
from feature_pipeline2 import features_pipeline2
from nltk.corpus import stopwords
from scipy import sparse
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
from nrclex import NRCLex
from models import DepressedBert
# from transformers import Trainer, BertForSequenceClassification, BertTokenizer
#from test_bert import inference_1

submit_logs_folder = os.path.join('data', 'submit')
if not os.path.exists(submit_logs_folder):
    os.makedirs(submit_logs_folder)

TEAM_TOKEN = f'v7PtOtt0pFUim9HbtrKqTiurdwRHgQR6Eh5sgZPT5xI'
GET_URL = f'https://erisk.irlab.org/challenge-service/getwritings/{TEAM_TOKEN}'
POST_URL = f'https://erisk.irlab.org/challenge-service/submit/{TEAM_TOKEN}'

# Global dictionary
full_texts_for_users = {}
current_text_for_user = {}
nicks_for_users = {}
dates_for_users = {}

# Vectorizer & Scaler
tfidf = pickle.load(open("models/tfidf_vectorizer_full_data.pkl", "rb"))
scaler = pickle.load(open("models/scaler_minmax_full.sav", "rb"))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# TODO: update the path if you have any issues
model_bert = BertForSequenceClassification.from_pretrained("models/checkpoint-15546")

tokenizer_thresh = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased")

# TODO: update the path if you have any issues
model_thresh = DepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=1)
state_dict = torch.load(os.path.join('logs', 'mental', 'mental-bert-base-uncased_2', 'best.pth'))
model_thresh.load_state_dict(state_dict["model"])
model_thresh.eval()


# Models
xgb_model = pickle.load(open('./bogdan_pickle_xgb_on_metadata.sav', 'rb'))
xgb_model_avg = pickle.load(open('./bogdan_pickle_xgb_on_metadata_avg.sav', 'rb'))
svc_model = pickle.load(open('models/svc_on_combined.pkl', 'rb'))
voting_model = pickle.load(open('models/voting_on_combined.pkl', 'rb'))
voting_text = pickle.load(open('models/voting_on_text.pkl', 'rb'))
# model = BertForSequenceClassification.from_pretrained("models/checkpoint-15546")

# Trainer
# trainer = Trainer(model=model, tokenizer=tokenizer)

# Preprocess functions
# Preprocessing stuff
nlp = spacy.load("en_core_web_sm")
stops = stopwords.words("english")
stops_dict = {stop: True for stop in stops}
all_emoji_emoticons = {**EMOTICONS_EMO, **UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
all_emoji_emoticons = {k: v.replace(":", "").replace("_", " ").strip() for k, v in all_emoji_emoticons.items()}

kp_all_emoji_emoticons = KeywordProcessor()
for k, v in all_emoji_emoticons.items():
    kp_all_emoji_emoticons.add_keyword(k, v)


def basic_preprocess_bert(text):
    text = re.sub(r'http\S+', ' ', text)
    return text


def full_preprocess_text(text, stopwords_removal=True):
    text = re.sub(r'http\S+', ' ', text)
    text = contractions.fix(text)
    text = re.sub(r'\[removed]', '', text)
    text = kp_all_emoji_emoticons.replace_keywords(text)
    VADER_dictionary = NRCLex(text).raw_emotion_scores.keys()
    if VADER_dictionary:
        for key in VADER_dictionary:
            text = text + " " + key
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    if stopwords_removal:
        text = ' '.join([word for word in text.split() if not stops_dict.get(word, False)])
    return text


# Models prediction functions

# Model trained on original metadata
def xgb_metadata_predict(user):
    text = full_texts_for_users[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline(dates, text)
    label = int(xgb_model.predict(np.array([features]))[0])
    score = float(xgb_model.predict_proba(np.array([features]))[0][1])
    return label, score


def voting_text_predict(user):
    text = full_texts_for_users[user]
    clean_text = full_preprocess_text(text)
    transform = tfidf.transform([clean_text])
    label = int(voting_text.predict(transform)[0])
    score = float(voting_text.predict_proba(transform)[0][1])
    return label, score


# Model trained on averaged metadata
def xgb_metadata_avg_predict(user):
    text = current_text_for_user.get(user, '')
    dates = dates_for_users.get(user, '')
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline2(dates, text)
    label = int(xgb_model_avg.predict(np.array([features]))[0])
    score = float(xgb_model_avg.predict_proba(np.array([features]))[0][1])
    return label, score


def svm_combined_predict(user):
    text = full_texts_for_users[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline(dates, text)
    clean_text = full_preprocess_text(text)
    transform = tfidf.transform([clean_text])
    meta_scaled = scaler.transform([features])
    meta_information = sparse.coo_matrix(meta_scaled)
    combined_text = sparse.hstack([transform, meta_information])
    label = int(svc_model.predict(combined_text)[0])
    score = float(svc_model.predict_proba(combined_text)[0][1])
    return label, score


def voting_combined_predict(user):
    text = full_texts_for_users[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline(dates, text)
    clean_text = full_preprocess_text(text)
    transform = tfidf.transform([clean_text])
    meta_scaled = scaler.transform([features])
    meta_information = sparse.coo_matrix(meta_scaled)
    combined_text = sparse.hstack([transform, meta_information])
    label = int(voting_model.predict(combined_text)[0])
    score = float(voting_model.predict_proba(combined_text)[0][1])
    return label, score


def bert_single_prediction(text):
    encodings = tokenizer(text,
                            max_length=512,
                            truncation=True,
                            padding=True,
                            return_tensors="pt")
    with torch.no_grad():
        outputs = model_bert(**encodings)
    preds = outputs.logits
    label = preds.argmax(-1).numpy()[0]
    proba = torch.nn.functional.softmax(preds, dim=1).detach().numpy()[0][1]
    return int(label), float(proba)

def bert_thresholded_single_prediction(text, threshold=55/99):
    encodings = tokenizer_thresh(text,
                            max_length=512,
                            truncation=True,
                            padding=True,
                            return_tensors="pt") 
    with torch.no_grad():
        output = model_thresh(**encodings)
        labels = (output.logits[:, 0] > threshold).int()
    label = labels[0].item()
    proba = output.logits[:, 0][0].item()
    return int(label), float(proba)

# Bert with Augmentation
def bert_prediction(user, single_pred):
    text = current_text_for_user.get(user, '')
    text = basic_preprocess_bert(text)

    if text == '':
        return 0, 0.0

    if len(text.split()) >= 450:
        batches_no = len(text.split()) // 450 + 1
        for i in range(batches_no):
            text = ' '.join(text.split()[(i * 450):(i + 1) * 450])

            label, proba = single_pred(text)
            if label == 1:
                return label, proba
        return 0, proba
    else:
        return single_pred(text)    


# BERT with Augmentation prediction
# def bert_prediction(user):
#     values = full_texts_for_users[user]
#     values = basic_preprocess_bert(values)
#     length_of_text = len(values.split())
#     if length_of_text >= 400:
#         batches_no = length_of_text // 400 + 1
#         for i in range(batches_no):
#             text = ' '.join(values.split()[(i * 400):(i + 1) * 400])
#             dataset = pd.DataFrame({'text': [text]})
#             custom_dataset = Dataset.from_pandas(dataset)
#             tokenized_text = custom_dataset.map(
#                 lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=512),
#                 batched=True)
#             tokenized_text = tokenized_text.remove_columns(['text'])
#             dec = np.argmax(trainer.predict(tokenized_text).predictions, axis=-1)[0]
#             if int(dec) == 1:
#                 return int(dec), float(
#                     tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][dec])
#         return int(dec), float(tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][dec])
#     else:
#         dataset = pd.DataFrame({'text': [values]})
#         custom_dataset = Dataset.from_pandas(dataset)
#         tokenized_text = custom_dataset.map(
#             lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=512), batched=True)
#         tokenized_text = tokenized_text.remove_columns(['text'])

#         decision = np.argmax(trainer.predict(tokenized_text).predictions, axis=-1)[0]
#         score = tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][decision]
#         return int(decision), float(score)


# user could be used to acces any data in the dictionaries above

full_texts_path = os.path.join(submit_logs_folder, 'full_texts_and_users.json')
if os.path.exists(full_texts_path):
    with open(full_texts_path, 'r') as infile:
        full_texts_for_users = json.load(infile)

nicks_path = os.path.join(submit_logs_folder, 'nicks_for_users.json')
if os.path.exists(nicks_path):
    with open(nicks_path, 'r') as infile:
        nicks_for_users = json.load(infile)

users_path = os.path.join(submit_logs_folder, 'dates_for_users.json')
if os.path.exists(users_path):
    with open(users_path, 'r') as infile:
        dates_for_users = json.load(infile)

#full_ct_models = [inference_1]
bert_pred = lambda user: bert_prediction(user, bert_single_prediction)
bert_thresh_pred = lambda user: bert_prediction(user, bert_single_prediction)

user_level_models = [voting_text_predict, xgb_metadata_avg_predict, svm_combined_predict, bert_pred, bert_thresh_pred]
# user_level_models = [voting_text_predict, xgb_metadata_avg_predict, svm_combined_predict, xgb_metadata_predict, bert_thresh_pred]

HEADERS = {
    'Content-type': 'application/json',
    'Accept': 'application/json'
}

step = 0
get_response = requests.get(GET_URL, headers=HEADERS)
print('Get request')
print(get_response)
answers = get_response.json()

should_continue = len(answers) > 0
start_time = time.perf_counter()

while should_continue:
    print(f'STEP: {step}')

    number = answers[0]["number"]
    # TODO: update the path if you have any issues
    with open('data/submit/number.out', 'w') as fout:
        fout.write(str(number))

    for answer in answers:
        redditor_str = str(answer['redditor'])
        answer_text = str(answer.get('title', '') + ' ' + answer.get('content', ''))

        current_text_for_user[redditor_str] = answer_text
        if redditor_str not in full_texts_for_users.keys():
            full_texts_for_users[redditor_str] = answer_text
        else:
            full_texts_for_users[redditor_str] = full_texts_for_users[redditor_str] + ' ' + \
                                                 answer_text

        if redditor_str not in nicks_for_users:
            nicks_for_users[redditor_str] = answer['nick']

        clean_date = answer['date'].split('.')[0]

        if redditor_str not in dates_for_users:
            dates_for_users[redditor_str] = [clean_date]
        else:
            dates_for_users[redditor_str].append(clean_date)

    # Just in case
    with open(f'{full_texts_path[:-5]}_{number}.json', 'w') as outfile:
        json.dump(full_texts_for_users, outfile)
    with open(f'{nicks_path[:-5]}_{number}.json', 'w') as outfile:
        json.dump(nicks_for_users, outfile)
    with open(f'{users_path[:-5]}_{number}.json', 'w') as outfile:
        json.dump(dates_for_users, outfile)

    with open(full_texts_path, 'w') as outfile:
        json.dump(full_texts_for_users, outfile)
    with open(nicks_path, 'w') as outfile:
        json.dump(nicks_for_users, outfile)
    with open(users_path, 'w') as outfile:
        json.dump(dates_for_users, outfile)

    run = 0
    for model in user_level_models:
        print(f'Run: {run}')
        results = []
        start_time = time.perf_counter()
        for user in full_texts_for_users.keys():
            # user can be used to get any information for the user from the dicts like the current text, full text etc
            label, score = model(user)
            results.append({
                'nick': nicks_for_users[user],
                'decision': label,
                'score': score
            })

            time_elapsed = time.perf_counter() - start_time
            if time_elapsed > 60:
                print(user)
                start_time = time.perf_counter()

        json_results = json.dumps(results)
        post_response = requests.post(f'{POST_URL}/{run}', data=json_results, headers=HEADERS)
        print('Post request done')
        print(post_response)
        run += 1

    '''for model in full_ct_models:
        print(f'Run: {run}')
        users_key = list(full_texts_for_users.keys())
        users_text = [basic_preprocess_bert(text) for text in full_texts_for_users.values()]

        results = []
        scores, labels = model(users_text)
        for user, label, score in zip(users_key, labels, scores):
            results.append({
                'nick': nicks_for_users[user],
                'decision': label,
                'score': score
            })

        json_results = json.dumps(results)
        post_response = requests.post(f'{POST_URL}/{run}', data=json_results, headers=HEADERS)
        print('Post request done')
        print(post_response)
        run += 1'''

    get_response = requests.get(GET_URL, headers=HEADERS)
    print('Get request done')
    print(get_response)
    answers = get_response.json()
    if len(answers) == 0:
        should_continue = False
    step += 1
