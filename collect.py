import xml.etree.ElementTree as etree
import json
import os
import pdb

from glob import glob
from __init__ import data_folder, data_json_path


def extract_xml_data(xlm_path, cases_folder, folder_label):
    with open(xlm_path) as fin:
        content = fin.read()
        tree = etree.fromstring(content)

    dataset = []
    for elem in tree.findall('WRITING'):
        dataset.append({
            "text": elem.find('TEXT').text,
            "date": elem.find('DATE').text,
            "info": elem.find('INFO').text,
            "source": cases_folder,
            "label": folder_label
        })
    return dataset

def main():
    dataset = []
    cases_folders = ["2017_cases", "2018_cases"]
    for cases_folder in cases_folders:
        for folder_label in ['neg', 'pos']:
            xml_re = os.path.join(data_folder, cases_folder, folder_label, '*.xml')
            xlm_paths = glob(xml_re)

            for i, xlm_path in enumerate(xlm_paths):
                dataset += extract_xml_data(xlm_path, cases_folder, folder_label)
                print(f"Collecting progress {xml_re}: {i}/{len(xlm_paths)}", end="\r")
            print()

    with open(data_json_path, 'w') as fout:
        json.dump(dataset, fout, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()