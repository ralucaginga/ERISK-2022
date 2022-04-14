import pandas as pd
import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from time import time
from tqdm import tqdm
import pickle
from time import time
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from glob import glob
from datetime import datetime
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
# from flashtext import KeywordProcessor
import nltk
import re
# import contractions
# from nrclex import NRCLex
# nltk.download('stopwords')
# nltk.download('punkt')

data_folder = 'dictionaries_depression'
antidepressant = os.path.join(data_folder, "antidepressants.txt")
three_grams = os.path.join(data_folder, "3-grams_suicide.txt")
five_grams = os.path.join(data_folder, "5-grams_suicide.txt")
over_generalization = os.path.join(data_folder, "over_generalization.txt")
psychoactive_drugs = os.path.join(data_folder, "psychoactive_drugs.txt")
unpleasant_feeling = os.path.join(data_folder, "unpleasant_feeling.txt")
nssi_words = os.path.join(data_folder, "nssi_words.txt")

with open(antidepressant, "r") as f:
    antidepressant_list = f.read().split("\n")

with open(three_grams, "r") as f:
    three_grams_list = f.read().split("\n")

with open(five_grams, "r") as f:
    five_grams_list = f.read().split("\n")

with open(over_generalization, "r") as f:
    over_generalization_list = f.read().split("\n")

with open(psychoactive_drugs, "r") as f:
    psychoactive_drug_list = f.read().split("\n")

with open(unpleasant_feeling, "r") as f:
    unpleasant_feeling_list = f.read().split("\n")

with open(nssi_words, "r") as f:
    nssi_list = f.read().split("\n")

temporal_past = ["yesterday", "last", "before", "ago", "past", "back", "earlier", "later"]

nlp = spacy.load("en_core_web_sm")


def basic_preprocess(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\[removed]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    text = text.replace(' ##', '')
    return text


def posting_time_level(dates):
    count = 0
    for date in dates:
        if date.time().hour >= 0 and date.time().hour <= 7:
            count += 1
    return count


def split_on_window(sequence, limit=4):
    results = []
    split_sequence = sequence.lower().split()
    iteration_length = len(split_sequence) - (limit - 1)
    max_window_indicies = range(iteration_length)
    for index in max_window_indicies:
        results.append(' '.join(split_sequence[index:index + limit]))
    return results


def count_depressive_terms(text):
    antidepress_count = len([sentence for sentence in text.split() if sentence.lower() in antidepressant_list])
    three_grams_count = len([sentence for sentence in split_on_window(text, 3) if sentence in three_grams_list])
    five_grams_count = len([sentence for sentence in split_on_window(text, 5) if sentence in five_grams_list])
    overgeneralization_count = len(
        [sentence for sentence in text.split() if sentence.lower() in over_generalization_list])
    psychoactive_count = len([sentence for sentence in text.split() if sentence.lower() in psychoactive_drug_list])
    unpleasant_feel_count = len([sentence for sentence in text.split() if sentence.lower() in unpleasant_feeling_list])
    nssi_count = len([sentence for sentence in text.split() if sentence.lower() in nssi_list])
    temporal_count = len([sentence for sentence in text.split() if sentence.lower() in temporal_past])

    return antidepress_count, three_grams_count, five_grams_count, overgeneralization_count, psychoactive_count, unpleasant_feel_count, nssi_count, temporal_count


def features_pipeline2(dates, text):
    words_count = len(text.split())
    punct_count = text.count('.') + text.count(',') + text.count(';') + text.count(':') + text.count('-')
    questions_count = text.count('?')
    exclamations_count = text.count('!')
    capitalized_count = sum(map(str.isupper, text.split()))

    tagged_doc = nlp(text)

    # Language Style
    try:
        adjective_count = len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == 'ADJ', tagged_doc))))
    except:
        adjective_count = 0

    try:
        verb_count = len(
            list(map(lambda w: w.text, filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'), tagged_doc))))
    except:
        verb_count = 0

    try:
        noun_count = len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == 'NOUN', tagged_doc)))) + len(
            list(map(lambda w: w.text, filter(lambda w: w.pos_ == 'PROPN', tagged_doc))))
    except:
        noun_count = 0

    try:
        adverb_count = len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADV", tagged_doc))))
    except:
        adverb_count = 0

    try:
        negation_count = len(list(map(lambda w: w.text,
                                      filter(lambda w: (w.pos_ == "PART" and w.morph.get("Polarity") == ["Neg"]),
                                             tagged_doc))))
    except:
        negation_count = 0

    try:
        formality_metric = (len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "NOUN", tagged_doc))))
                            + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "PROPN", tagged_doc))))
                            + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADJ", tagged_doc))))
                            + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADP", tagged_doc))))
                            + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "DET", tagged_doc))))
                            - len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "PRON", tagged_doc))))
                            - len(
                    list(map(lambda w: w.text, filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'), tagged_doc))))
                            - len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADV", tagged_doc))))
                            - len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "INTJ", tagged_doc))))
                            + 100) / 2
    except:
        formality_metric = 0

    try:
        trager_coefficient = len(
            list(map(lambda w: w.text, filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'), tagged_doc)))) / len(
            list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADJ", tagged_doc))))
    except:
        trager_coefficient = 0

    try:
        readiness_to_action_coefficient = len(
            list(map(lambda w: w.text, filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'), tagged_doc)))) / (
                                                      len(list(map(lambda w: w.text,
                                                                   filter(lambda w: w.pos_ == "NOUN", tagged_doc))))
                                                      + len(list(map(lambda w: w.text,
                                                                     filter(lambda w: w.pos_ == "PROPN", tagged_doc)))))
    except:
        readiness_to_action_coefficient = 0

    try:
        aggressiveness_coefficient = len(list(map(lambda w: w.text, filter(
            lambda w: (w.pos_ == "VERB" and w.morph.get("VerbForm") == ['Part']), tagged_doc)))) / words_count
    except:
        aggressiveness_coefficient = 0

    try:
        activity_index = len(
            list(map(lambda w: w.text, filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'), tagged_doc)))) / (
                                     len(list(map(lambda w: w.text,
                                                  filter(lambda w: (w.pos_ == 'AUX') | (w.pos_ == 'VERB'),
                                                         tagged_doc))))
                                     + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADJ", tagged_doc))))
                                     + len(list(map(lambda w: w.text, filter(lambda w: w.pos_ == "ADP", tagged_doc)))))
    except:
        activity_index = 0

    # User behaviour
    time_level = posting_time_level(dates)

    # Self-Preoccupation
    try:
        first_person_pron_count = len(list(
            map(lambda w: w.text, filter(lambda w: (w.pos_ == "PRON" and w.morph.get("Person") == ['1']), tagged_doc))))
    except:
        first_person_pron_count = 0

    # Reminiscicence & Sentiment
    antidepress_count, three_grams_count, five_grams_count, overgeneralization_count, psychoactive_count, unpleasant_feel_count, nssi_count, temporal_count = count_depressive_terms(
        text)



    list_of_features = [adjective_count / words_count, verb_count  / words_count, noun_count  / words_count, adverb_count  / words_count,
       negation_count / words_count, formality_metric, readiness_to_action_coefficient,
       aggressiveness_coefficient, activity_index, time_level,
       first_person_pron_count / words_count, antidepress_count / words_count, three_grams_count / words_count,
       five_grams_count / words_count, overgeneralization_count / words_count, psychoactive_count / words_count,
       unpleasant_feel_count / words_count, nssi_count / words_count, temporal_count / words_count, punct_count / words_count,
       questions_count / words_count, exclamations_count / words_count / words_count, capitalized_count / words_count]

    return list_of_features


# %%
### Testing
values = [
    "I have paranoia and depression. I have anxiety and I hate this life",
    "I go to therapy. I have paranoia and go to psychologist",
    "I like dogs, but I am hurting myself and biting my nails because of frustration",
    "I was in a dark place and needed reassurance from my significant other that I was loved and wanted, and I then apologized for wanting and needing that. This was his response.",
    "I had an early morning panic attack and texted my friend at 3:30 a.m… she was beyond awesome and helped me out even though I woke her up",
    "Ukraine"]
dates = [["2021-02-20 12:00:20", "2021-02-21 00:20:12"],
         ["2021-02-20 00:00:20"],
         ["2021-02-10 12:00:20"],
         ["2021-02-10 12:00:20"],
         ["2021-02-10 12:00:20"],
         ["2021-02-10 12:00:20"]]

clean_text = []
features = []
all_dates = []
for text in values:
    clean_text.append(basic_preprocess(text))
    all_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for page in dates for date in page]
    features.append(features_pipeline2(all_dates, text))
# %%
