import xml.etree.ElementTree as etree
import json
import os
import numpy as np
import pdb

from glob import glob
from __init__ import data_folder, data_json_path
from datetime import datetime


def extract_xml_data(xml_path, cases_folder, label_id):
    with open(xml_path) as fin:
        content = fin.read()
        tree = etree.fromstring(content)

    dataset = []
    for elem in tree.findall('WRITING'):
        text = elem.find('TITLE').text + elem.find('TEXT').text
        dataset.append({
            "text": text.strip(),
            "date": elem.find('DATE').text.strip(),
            "info": elem.find('INFO').text.strip(),
            "source": cases_folder,
            "label": label_id
        })
    return dataset

def main():
    dataset = {}
    cases_folders = ["2017_cases", "2018_cases"]
    for cases_folder in cases_folders:
        for label_id, folder_label in enumerate(['neg', 'pos']):
            xml_re = os.path.join(data_folder, cases_folder, folder_label, '*.xml')
            xlm_paths = glob(xml_re)

            first_date = None
            last_date = None
            posts_per_user = []
            time_spans_days = []
            n_posts = 0
            for i, xml_path in enumerate(xlm_paths):
                xml_subject = os.path.basename(xml_path)
                xml_data = extract_xml_data(xml_path, cases_folder, label_id)
                dataset.update({
                    xml_subject: xml_data
                })
                n_posts += len(xml_data)
                posts_per_user.append(len(xml_data))

                first_date = datetime.strptime(xml_data[0]["date"], "%Y-%m-%d %H:%M:%S")
                last_date = datetime.strptime(xml_data[-1]["date"], "%Y-%m-%d %H:%M:%S")

                time_span_ = last_date - first_date
                time_span_days = np.abs(time_span_.days)
                time_spans_days.append(time_span_days)

                print(f"Collecting progress {xml_re}: {i}/{len(xlm_paths)}", end="\r")
            print()
            
            mean_post_per_user = np.mean(posts_per_user)
            mean_time_span_days = np.mean(time_spans_days)
            print(f"{xml_re} finished with {len(xlm_paths)} users, {n_posts} total posts and mean_post_per_user {mean_post_per_user} and mean_time_span_days =  {mean_time_span_days}" )

    with open(data_json_path, 'w') as fout:
        json.dump(dataset, fout, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()

# from sklearn.model_selection import TimeSeriesSplit
# https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1