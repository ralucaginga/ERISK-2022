import os
import torch
import torch.nn as nn
import json
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

from __init__ import train_json, dev_json, test_dev_json, test_json_path
from models import DepressedBert

tokenizer = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased")


class EriskDataset(Dataset):
    def __init__(self, scope):
        if scope == "train":
            json_path = train_json
        elif scope == "dev":
            json_path = dev_json
        elif scope == "test":
            json_path = test_dev_json
        
        self.tokenizer = tokenizer
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



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def export_probas_and_labels(model, test_dataloader, exp_dir):
    real_labels = []
    out_labels = []
    out_probas = []
    for i, batch in enumerate(test_dataloader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        real_labels += labels.tolist()

        with torch.no_grad():
            output = model(input_ids)
            output = nn.Sigmoid()(output[0])
            result_labels = (output[:, 0] >= 0.5).int()
            result_proba = output[:, 0]

        out_labels += result_labels.tolist()
        out_probas += result_proba.tolist()
        
        if i % 100 == 0:
            print(f'Testing progress {i}/{len(test_dataloader)}')

    all_labels = np.array([out_labels, real_labels])
    labels_path = os.path.join(exp_dir, "dev_labels.npy")
    np.save(labels_path, all_labels)

    all_probas = np.array([out_probas, real_labels])
    probas_path = os.path.join(exp_dir, "dev_probas.npy")
    np.save(probas_path, all_probas)


def inference(model, text, threshold=54/99):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, padding='max_length', max_length=150)
    token_ids = token_ids.to(device)
    with torch.no_grad():
        output = model(token_ids)
        labels = (output.logits[:, 0] > threshold).int()
    return labels[0].item()




def main():
    exp_dir = os.path.join('logs', 'mental', 'mental-bert-base-uncased_2')
    model_path = os.path.join(exp_dir, 'best.pth')

    model = DepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=1).to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # test_dataset = EriskDataset('test')
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    # export_probas_and_labels(model, test_dataloader, exp_dir)

    with open(test_json_path, 'r') as fin:
        test_set = json.load(fin)

    for entry in test_set:
        label = inference(model, entry["text"])
        entry["predicted"] = label

    with open(test_json_path, 'w') as fout:
        json.dump(test_set, fout, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()