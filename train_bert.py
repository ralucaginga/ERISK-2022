# %%
import os
data_folder = 'data'

full_df_path = os.path.join(data_folder, 'full_dataframe.csv')

train_json = os.path.join(data_folder, 'train_.json')
dev_json = os.path.join(data_folder, 'dev_.json')
logs_folder = 'logs'

# %%
# !ls {data_folder}

# %%
import torch
import torch.nn as nn
import json
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from glob import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

# %%
# import pdb
# from tqdm import tqdm

# import pandas as pd
# df = pd.read_csv('data/full_dataframe.csv')

# poz_number = (df['label'] == 1).sum()
# neg_indices = np.where(df['label'] == 0)[0]
# rand_indices = np.random.randint(len(neg_indices), size=poz_number)

# poz_indices = np.where(df['label'] == 1)[0]
# neg_indices = neg_indices[rand_indices]
# sub_indices = np.concatenate([poz_indices, neg_indices])
# sub_df = df.iloc[sub_indices].reset_index(drop=True)


# texts = sub_df['text'].tolist()
# labels = sub_df['label'].tolist()

# max_len = 150
# tokenizer = BertTokenizer.from_pretrained("mental/mental-bert-base-uncased")

# split_data = []
# for text, label in tqdm(zip(texts, labels)):
#     all_tokens = tokenizer.tokenize(text)
#     n_tokens = len(all_tokens)
    
#     for i in range(0, n_tokens, max_len):
#         current_tokens = all_tokens[i:i+max_len]
#         current_text = ' '.join(current_tokens)
#         current_text = current_text.replace(' ##', '')
        
#         split_data.append({
#             "text": current_text, 
#             "label": label
#         })
        
# from sklearn.model_selection import train_test_split
# train_data, dev_data, _, _ = train_test_split(split_data, np.zeros(len(split_data)), test_size=0.2, random_state=101)

# with open('data/train_.json', 'w') as fout:
#     json.dump(train_data, fout, indent=4, sort_keys=True)

# with open('data/dev_.json', 'w') as fout:
#     json.dump(dev_data, fout, indent=4, sort_keys=True)

# pdb.set_trace()

# %%
class EriskDataset(Dataset):
    def __init__(self, scope):
        if scope == "train":
            json_path = train_json
        elif scope == "dev":
            json_path = dev_json
        
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

# %%
class DepressedBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)
        )

# %%
EriskDataset('train')[0]

# %%
with open(train_json, 'r') as fin:
    json_list = json.load(fin)
    
n_positives = 0
for entry in json_list:
    n_positives += entry['label'] == 1

pos_weight = n_positives / len(json_list)
# Version 1: 
# pos_weight = n_positives / len(json_list)
# pos_weight , 1 -  pos_weight
# Version 2 (reverse importance):
# 1 - pos_weight, pos_weight
# Version 3:
# new parameters for training
# https://iq.opengenus.org/binary-text-classification-bert/
# internal BCE

# %%
class BCEWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_weight = 1 - pos_weight
        self.neg_weight = pos_weight
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, output, label):
        output = self.sigmoid(output)
        loss = - self.pos_weight * (label * torch.log(output)) - \
                 self.neg_weight * ((1 - label) * torch.log(1 - output))
        loss = loss.mean(axis=0)
        return loss

# %%
import pdb
def get_lr(optimizerr):
    for param_group in optimizerr.param_groups:
        return param_group['lr']

def train_epoch(epoch, model, dataloaders, optimizerr, lr_sched, writer, exp_dir, best_loss):
    for phase in ['train', 'dev']:
        epoch_losses = []

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for i, batch in enumerate(dataloaders[phase]):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.float().to(device)

            optimizerr.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                output = model(input_ids, labels=labels)
                loss = output.loss
#                 output = output[0][:, 0]
#                 loss = BCEWeightedLoss()(output, labels)
#                 output = nn.Sigmoid()(output)
#                 loss = nn.BCELoss()(output, labels)

                if phase == 'train':
                    loss.backward()
                    optimizerr.step()

            epoch_losses.append(loss.item())
            average_loss = np.mean(epoch_losses)
            lr = get_lr(optimizerr)

            if (i + 1) % 10 == 0:
                loading_percentage = int(100 * (i+1) / len(dataloaders[phase]))
                print(f'{phase}ing epoch {epoch}, iter = {i+1}/{len(dataloaders[phase])} ' + \
                    f'({loading_percentage}%), loss = {loss}, average_loss = {average_loss} ' + \
                    f'learning rate = {lr}')
                
        if phase == 'dev' and average_loss < best_loss:
            best_loss = average_loss
            
            torch.save({
                    'model': model.state_dict(),
                    'optimizerrizer': optimizerr.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                    'epoch': epoch,
                    'dev_loss': best_loss
                }, os.path.join(exp_dir, 'best.pth'))
            
        if phase == 'train':
            metric_results = {
                'train_loss': average_loss
            }

            writer.add_scalar('Train/Loss', average_loss, epoch)
            writer.flush()

        if phase == 'dev':
            val_results = {
                'dev_loss': average_loss
            }

            metric_results.update(val_results)
            writer.add_scalar('Dev/Loss', average_loss, epoch)
            writer.flush()

            try:
                lr_sched.step()
            except:
                lr_sched.step(average_loss)
        
        print()
    return best_loss

# %%
def is_inner_parameter(name):
    if name.startswith('bert.encoder.layer') and int(name.split('.')[3]) >= 10:
        return False
    if name.startswith('bert.pooler'):
        return False
    if name.startswith('classifier'):
        return False
    return True

# %%
config = {
    "model_name": "mental/mental-bert-base-uncased",
    "batch_size": 4,
    "optimizer": "AdamW",
    "n_epochs": 10,
    "n_target": 1
}
batch_size = config["batch_size"]
train_dataset = EriskDataset('train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
                                    shuffle=True, num_workers=0)
dev_dataset = EriskDataset('dev')
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, \
                                    shuffle=False, num_workers=0)
dataloaders = {
    "train": train_dataloader,
    "dev": dev_dataloader
}

model = DepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=config["n_target"]).to(device)
for parameter in model.bert.parameters():
    parameter.requires_grad = False

if config["optimizer"] == "AdamW":
    optimizerr = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
lr_sched = ReduceLROnPlateau(optimizerr, factor = 0.1, patience = 3, mode = 'min')

model_name = config['model_name']
experiments_re = os.path.join(logs_folder, f'{model_name}*')
n_experments = len(glob(experiments_re))
exp_dir = os.path.join(logs_folder, f'{model_name}_{n_experments}')

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
writer = SummaryWriter(os.path.join(exp_dir, 'runs'))

config_path = os.path.join(exp_dir, 'config.json')
with open(config_path, 'w') as fout:
    json.dump(config, fout, indent=4, sort_keys=True)

# %%
best_loss = 5000
for epoch in range(config["n_epochs"]):
    best_loss = train_epoch(epoch, model, dataloaders, optimizerr, lr_sched, writer, exp_dir, best_loss)

# %%
# exp_dir = os.path.join('logs', 'mental', 'mental-bert-base-uncased_3')
model_path = os.path.join(exp_dir, 'best.pth')
state_dict = torch.load(model_path)
model.load_state_dict(state_dict["model"])
model.eval()


real_labels = []
out_labels = []
for i, batch in enumerate(dev_dataloader):
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    real_labels += labels.tolist()

    with torch.no_grad():
        output = model(input_ids)
        output = nn.Sigmoid()(output[0])
        result_labels = (output[:, 0] >= 0.5).int()
    out_labels += result_labels.tolist()
    
    if i % 100 == 0:
        print(f'Testing progress {i}/{len(dev_dataloader)}')

# %%
all_labels = np.array([out_labels, real_labels])
labels_path = os.path.join(exp_dir, "dev_labels.npy")
np.save(labels_path, all_labels)

# %%
# !ls {exp_dir}

# %%



# %%



