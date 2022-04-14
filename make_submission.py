import json
import pickle
import tensorflow as tf
import re
import spacy
import pandas as pd
import contractions
import numpy as np
import requests

from datetime import datetime
from feature_pipeline import features_pipeline
from feature_pipeline2 import features_pipeline2
from nltk.corpus import stopwords
from scipy import sparse
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
from nrclex import NRCLex
from datasets import Dataset
from transformers import Trainer, BertForSequenceClassification, BertTokenizer

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

# Models
xgb_model = pickle.load(open('./bogdan_pickle_xgb_on_metadata.sav', 'rb'))
xgb_model_avg = pickle.load(open('./bogdan_pickle_xgb_on_metadata_avg.sav', 'rb'))
svc_model = pickle.load(open('models/svc_on_combined.pkl', 'rb'))
voting_model = pickle.load(open('models/voting_on_combined.pkl', 'rb'))
model = BertForSequenceClassification.from_pretrained("models/checkpoint-15546")

# Trainer
trainer = Trainer(model=model,
                  tokenizer=tokenizer)

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
    score = float(xgb_model.predict_proba(np.array([features]))[0][label])
    return label, score


# Model trained on averaged metadata
def xgb_metadata_avg_predict(user):
    text = current_text_for_user[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline2(dates, text)
    label = int(xgb_model_avg.predict(np.array([features]))[0])
    score = float(xgb_model.predict_proba(np.array([features]))[0][label])
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
    score = float(svc_model.predict_proba(combined_text)[0][label])
    print(score)
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
    score = float(voting_model.predict_proba(combined_text)[0][label])
    print(score)
    return label, score


# BERT with Augmentation prediction
def bert_prediction(user):
    values = full_texts_for_users[user]
    values = basic_preprocess_bert(values)
    length_of_text = len(values.split())
    if length_of_text >= 400:
        batches_no = length_of_text // 400 + 1
        for i in range(batches_no):
            text = ' '.join(values.split()[(i * 400):(i + 1) * 400])
            dataset = pd.DataFrame({'text': [text]})
            custom_dataset = Dataset.from_pandas(dataset)
            tokenized_text = custom_dataset.map(
                lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=512),
                batched=True)
            tokenized_text = tokenized_text.remove_columns(['text'])
            dec = np.argmax(trainer.predict(tokenized_text).predictions, axis=-1)[0]
            if int(dec) == 1:
                return int(dec), float(
                    tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][dec])
        return int(dec), float(tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][dec])
    else:
        dataset = pd.DataFrame({'text': [values]})
        custom_dataset = Dataset.from_pandas(dataset)
        tokenized_text = custom_dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=512), batched=True)
        tokenized_text = tokenized_text.remove_columns(['text'])

        decision = np.argmax(trainer.predict(tokenized_text).predictions, axis=-1)[0]
        score = tf.math.softmax(trainer.predict(tokenized_text).predictions, axis=-1).numpy()[0][decision]
        return int(decision), float(score)


# user could be used to acces any data in the dictionaries above



# TODO: Add your prediction functions here
# TODO: To be modified the functions that are fast
models = [xgb_metadata_avg_predict, svm_combined_predict, voting_combined_predict, bert_prediction] * 1
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

while should_continue:
    print(f'STEP: {step}')
    for answer in answers:
        answer_text = answer['title'] + ' ' + answer['content']

        current_text_for_user[answer['redditor']] = answer_text
        if answer['redditor'] not in full_texts_for_users.keys():
            full_texts_for_users[answer['redditor']] = answer_text
        else:
            full_texts_for_users[answer['redditor']] = full_texts_for_users[answer['redditor']] + ' ' + \
                                                       answer_text

        if answer['redditor'] not in nicks_for_users:
            nicks_for_users[answer['redditor']] = answer['nick']

        clean_date = answer['date'].split('.')[0]

        if answer['redditor'] not in dates_for_users:
            dates_for_users[answer['redditor']] = [clean_date]
        else:
            dates_for_users[answer['redditor']].append(clean_date)

    # Just in case
    with open('full_texts_and_users.json', 'w') as outfile:
        json.dump(full_texts_for_users, outfile)
    with open('nicks_for_users.json', 'w') as outfile:
        json.dump(nicks_for_users, outfile)
    with open('dates_for_users.json', 'w') as outfile:
        json.dump(dates_for_users, outfile)

    for run in range(0, 1):
        print(f'Run: {run}')
        results = []
        users = full_texts_for_users.keys()
        for user in full_texts_for_users.keys():
            # user can be used to get any information for the user from the dicts like the current text, full text etc
            label = models[run](user)
            if label != 0:
                # SHould probably comment this
                print("One found")
                print(full_texts_for_users[user])
            nick = nicks_for_users[user]
            results.append({
                'nick': nick,
                'decision': label,
                'score': label
            })
        json_results = json.dumps(results)
        post_response = requests.post(f'{POST_URL}/{run}', data=json_results, headers=HEADERS)
        print('Post request done')
        print(post_response)
    get_response = requests.get(GET_URL, headers=HEADERS)
    print('Get request done')
    print(get_response)
    answers = get_response.json()
    if len(answers) == 0:
        should_continue = False
    step += 1
