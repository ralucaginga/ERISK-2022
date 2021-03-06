import os
import torch
import torch.nn as nn
import json
import numpy as np
import time
import pdb

from transformers import BertTokenizer

from __init__ import test_json_path
from models import DepressedBert
from transformers import BertTokenizer, BertForSequenceClassification

softmax = nn.Softmax(dim=-1)

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

def inference(model, texts, threshold=55/99, batch_size=8):
    all_probas = []
    all_labels = []
    n_texts = len(texts)
    
    for start in range(0, n_texts, batch_size):
        start_time = time.perf_counter()
        
        token_dict = tokenizer(texts[start: start + batch_size], return_tensors='pt', \
                              truncation=True, padding='max_length', max_length=512, \
                              return_token_type_ids=False, return_attention_mask=False)
        token_ids = token_dict['input_ids'].to(device)        
        with torch.no_grad():
            output = model(token_ids)
            probas = output.logits[:, 0]
            labels = (output.logits[:, 0] > threshold).int() 

        all_probas.extend(probas.tolist())
        all_labels.extend(labels.tolist())
        del token_ids
        
        time_elapsed = time.perf_counter() - start_time
        print(f"Position {start}/{n_texts} ended in {time_elapsed} seconds")
    return all_probas, all_labels

# def inference_2(model, texts, batch_size=8):
#     all_probas = []
#     all_labels = []
#     n_texts = len(texts)
    
#     for start in range(0, n_texts, batch_size):
#         start_time = time.perf_counter()
        
#         token_dict = tokenizer(texts[start: start + batch_size], return_tensors='pt', \
#                               truncation=True, padding='max_length', max_length=512, \
#                               return_token_type_ids=False, return_attention_mask=False)
#         token_ids = token_dict['input_ids'].to(device)        
#         with torch.no_grad():
#             output = model(token_ids)
#             output = softmax(output.logits)
#             labels = torch.argmax(output, axis=-1)

#         all_probas.extend(output[:, 1].tolist())
#         all_labels.extend(labels.tolist())
#         del token_ids
        
#         time_elapsed = time.perf_counter() - start_time
#         print(f"Position {start}/{n_texts} ended in {time_elapsed} seconds")
#     # pdb.set_trace()
#     return all_probas, all_labels


def get_model_by_exp_dir(model_path):
    model = DepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=1)#.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    return model

model = get_model_by_exp_dir(os.path.join('logs', 'mental', 'mental-bert-base-uncased_2', 'best.pth'))
inference_1 = lambda texts: inference(model, texts)
tokenizer = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased")

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("models/checkpoint-15546").to(device)
# model.eval()
# inference_1 = lambda texts: inference_2(model, texts)


def main():
    exp_dir = os.path.join('logs', 'mental', 'mental-bert-base-uncased_5')
    model_path = os.path.join(exp_dir, 'best.pth')

    model = get_model_by_exp_dir(model_path)

    # test_dataset = EriskDataset('test')
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    # export_probas_and_labels(model, test_dataloader, exp_dir)

    with open(test_json_path, 'r') as fin:
        test_set = json.load(fin)

    for entry in test_set:
        label = inference_single(model, entry["text"])
        entry["predicted"] = label

    with open(test_json_path, 'w') as fout:
        json.dump(test_set, fout, indent=4, sort_keys=True)

if __name__ == "__main__":
    # print(inference_1('If you want to make money look at other options than precious metals. If you want silver in the UK look at the group orders by backyard bullion on the silver forum!'))
    main()