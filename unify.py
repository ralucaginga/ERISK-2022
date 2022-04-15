
json_path1 = 'data/submit/full_texts_and_users_bkp1.json'
json_path2 = 'data/submit/full_texts_and_users_bkp2.json'
merged_path = 'data/submit/full_texts_and_users.json'
import json
import pdb

with open(json_path1, 'r') as infile:
    history = json.load(infile)

with open(json_path2, 'r') as infile:
    latter = json.load(infile)

for key in latter.keys():
    history[key] = history.get(key, '') + ' ' +  latter.get(key)

with open(merged_path, 'w') as fout:
    json.dump(history, fout, indent=4)