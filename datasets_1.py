import torch
import json

from torch.utils.data import Dataset
from __init__ import train_json, dev_json, test_dev_json
from transformers import BertTokenizer


class EriskDataset(Dataset):
    def __init__(self, scope):
        if scope == "train":
            json_path = train_json
        elif scope == "dev":
            json_path = dev_json
        elif scope == "test":
            json_path = test_dev_json
        
        self.tokenizer = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased")
        self.max_len = 150
        with open(json_path, 'r') as fin:
            self.json_list = json.load(fin)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, index):
        entry = self.json_list[index]
        text = entry["text"]
        input_ids = self.tokenizer.encode(text, return_tensors='pt', truncation=True, \
                                            padding='max_length', max_length=self.max_len)[0]
        label = entry["label"]
        return input_ids, torch.tensor(label)