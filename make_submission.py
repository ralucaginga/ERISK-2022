import json
from datetime import datetime
import pickle

import numpy as np
import requests
from feature_pipeline import features_pipeline

TEAM_TOKEN = f'v7PtOtt0pFUim9HbtrKqTiurdwRHgQR6Eh5sgZPT5xI'
GET_URL = f'https://erisk.irlab.org/challenge-service/getwritings/{TEAM_TOKEN}'
POST_URL = f'https://erisk.irlab.org/challenge-service/submit/{TEAM_TOKEN}'


# Global dictionary
full_texts_for_users = {}
current_text_for_user = {}
nicks_for_users = {}
dates_for_users = {}


xgb_model = pickle.load(open('./bogdan_pickle_xgb_on_metadata.sav', 'rb'))


# user could be used to acces any data in the dictionaries above
def xgb_metadata_predict(user):
    text = full_texts_for_users[user]
    dates = dates_for_users[user]
    dates = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in dates]
    features = features_pipeline(dates, text)
    label = int(xgb_model.predict(np.array([features]))[0])
    return label


# TODO: Add your prediction functions here
models = [xgb_metadata_predict] * 5
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

    for run in range(0, 5):
        results = []
        for user in full_texts_for_users.keys():
            # user can be used to get any information for the user from the dicts like the current text, full text etc
            label = models[run](user)
            if label != 0:
                print("One found")
            nick = nicks_for_users[user]
            results.append({
                'nick': nick,
                'decision': label,
                'score': label
            })
        json_results = json.dumps(results)
        print(f'Run: {run}')
        post_response = requests.post(f'{POST_URL}/{run}', data=json_results, headers=HEADERS)
        print(post_response)
    get_response = requests.get(GET_URL, headers=HEADERS)
    print('Get request')
    print(get_response)
    answers = get_response.json()
    if len(answers) == 0:
        should_continue = False
    step += 1
