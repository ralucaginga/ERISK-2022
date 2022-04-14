import json
from datetime import datetime
import pickle
import pdb
import time

import numpy as np
import requests
# from feature_pipeline import features_pipeline
from feature_pipeline2 import features_pipeline2
from test_bert import inference_1

TEAM_TOKEN = f'v7PtOtt0pFUim9HbtrKqTiurdwRHgQR6Eh5sgZPT5xI'
GET_URL = f'https://erisk.irlab.org/challenge-service/getwritings/{TEAM_TOKEN}'
POST_URL = f'https://erisk.irlab.org/challenge-service/submit/{TEAM_TOKEN}'


# Global dictionary
full_texts_for_users = {}
current_text_for_user = {}
nicks_for_users = {}
dates_for_users = {}


xgb_model = pickle.load(open('./bogdan_pickle_xgb_on_metadata.sav', 'rb'))
xgb_model_avg = pickle.load(open('./bogdan_pickle_xgb_on_metadata_avg.sav', 'rb'))


# user could be used to acces any data in the dictionaries above

# Model trained on original metadata
def xgb_metadata_predict(user):
    text = full_texts_for_users[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline(dates, text)
    label = int(xgb_model.predict(np.array([features]))[0])
    return label

# Model trained on averaged metadata
def xgb_metadata_avg_predict(user):
    text = current_text_for_user[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline2(dates, text)
    label = int(xgb_model_avg.predict(np.array([features]))[0])
    return label



# TODO: Add your prediction functions here
user_level_models = [] # [xgb_metadata_avg_predict]
full_ct_models = [inference_1] 
HEADERS = {
    'Content-type':'application/json',
    'Accept':'application/json'
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
    with open('data/submit/full_texts_and_users.json', 'w') as outfile:
        json.dump(full_texts_for_users, outfile)
    with open('data/submit/nicks_for_users.json', 'w') as outfile:
        json.dump(nicks_for_users, outfile)
    with open('data/submit/dates_for_users.json', 'w') as outfile:
        json.dump(dates_for_users, outfile)

    run = 0
    for model in full_ct_models:
        print(f'Run: {run}')
        users_key = list(full_texts_for_users.keys())
        users_text = list(full_texts_for_users.values())
        
        results = []
        labels, score = model(users_text)
        for user, label, score in zip(users_key, labels, score):
            results.append({
                'nick': nicks_for_users[user],
                'decision': label,
                'score': score
            })

        json_results = json.dumps(results)
        post_response = requests.post(f'{POST_URL}/{run}', data=json_results, headers=HEADERS)
        print('Post request done')
        print(post_response)
        
        run += 1


    for model in user_level_models:
        print(f'Run: {run}')
        results = []
        for user in full_texts_for_users.keys():
            # user can be used to get any information for the user from the dicts like the current text, full text etc
            label = model(user)
            # if label != 0:
            #     # SHould probably comment this
            #     print("One found")
            #     print(full_texts_for_users[user])
            results.append({
                'nick': nicks_for_users[user],
                'decision': label,
                'score': label
            })
            time_elapsed = time.perf_counter() - start_time 
            # 169 useri / minut 
            # 2289 useri -> 13.5 minute
            if time_elapsed > 60:
                pdb.set_trace()

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
