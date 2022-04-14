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

def inference_single(model, text, threshold=52/99):
    token_dict = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512, \
                              return_token_type_ids=False, return_attention_mask=False)
    token_ids = token_dict['input_ids'].to(device)        
    with torch.no_grad():
        output = model(token_ids)
        labels = (output.logits[:, 0] > threshold).int()
    return labels[0].item()

def inference(model, texts, threshold=55/99, batch_size=8):
    all_probas = []
    all_labels = []
    n_texts = len(texts)
    
    for start in range(0, n_texts, batch_size):
        start_time = time.perf_counter()
        
        token_dict = tokenizer(texts[start: start + batch_size], return_tensors='pt', \
                              truncation=True, padding='max_length', max_length=400, \
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

def inference_2(model, texts, batch_size=8):
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
            output = softmax(output.logits)
            labels = torch.argmax(output, axis=-1)

        all_probas.extend(output[:, 1].tolist())
        all_labels.extend(labels.tolist())
        del token_ids
        
        time_elapsed = time.perf_counter() - start_time
        print(f"Position {start}/{n_texts} ended in {time_elapsed} seconds")
    pdb.set_trace()
    return all_probas, all_labels


def get_model_by_exp_dir(model_path):
    # check here
    model = DepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=1).to(device)
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
    # main()
    inference_1([
        "         Let's get real: they are not gonna add them. They already said they don't like game ending/changing killstreaks :(",
          "         (A bit long for ELI5, so apologies) Each citizen needs 2 food to survive. If you have food yields (from buildings, working food tiles, or trade routes) greater than (2)(Population), the extra food gets added to your food basket each turn. Once this basket is filled, you get an extra citizen! This citizen can now work a new tile, giving you more food/production/gold, and the cycle continues. After each population growth, the food basket is empty and also gets a little bit bigger, which is why your cities grow much faster when small, and much slower when large. This also explains why some policies/buildings improve +growth, +excess food, or +food leftover (these are all different concepts). Your production is directly tied to your citizens (who work production tiles or act as specialists) and your buildings. If you have 1 citizen, and he works a mine for 3 production, each turn you can produce 3 hammers. If a building costs 30 hammers, it will take 10 turns. If your population grows to 2, and now you can work 2 mines, you produce 6 hammers per turn, and your building is now built in 5 turns. As you can easily see, the more citizens you have to provide production (either by working tiles or as specialists), the greater your production each turn is, and the faster you can build buildings. The actual formulas vary greatly based on the tiles you can work, the improvements your workers make, and the buildings in the city. An easy way to understand this mechanic is, when your city first grows to Population 2, open up your city management screen (by clicking on the city). Then click Citizen Management in the upper right corner, and move your citizen around to different tiles. You will see direct changes to however long the building in your production queue has left, and you will see changes to 'Turns to Next Citizen.' Now take what you know for how 1 citizen moved around changes this, and multiply it by your entire populace for later in the game. Each game has a trade-off decision you have to make. Typically 'production-focused' tiles like mines will help you build early buildings much faster, but your cities will grow slower (and thus you have fewer citizens to work tiles). Alternatively, you can work farms to grow your city quickly, but your buildings will take much longer to build (until you have enough citizens to work production tiles in addition to food tiles). Since population drives several other key resources (like gold from trade routes and science), you generally want to do everything possible to get your cities as large as your happiness allows. Typically your production keeps up as your cities grow (assuming you settled near a few hills or forests), but you may want to manually adjust your citizens every so often to make sure your production isn't too low. Alternatively, if your cities aren't growing (and you want them to), you may need to re-assign citizens to food tiles at the expense of production.",
    "    Update 3:18 AM, Big Ten logo is getting blued out     As title said, Big Ten logo is getting fucked up. See if you can do anything to help. Also, we're trying to build up a megathread at UIUC, found [here](https://www.reddit.com/r/UIUC/comments/6354dh/megathread_for_rplace/) if you want to help out",
         "    blender with rust source bindings     is it possible to develop addons in rustlang? and intergrate it with blender via python?",
         "    A quick question about the physics engine.     I searched around a bit, and wasn't able to find an answer to this question, though maybe I'm just bad at wording things in search engines. A while back, presumably after a past update, my physics started to get a bit weird. Everything seemed to lose most of its mass, to the point that I'm beginning to suspect that Appalachia is actually on Mars. Ragdolls, thrown grenades and mines, and basically anything else physics driven has become incredibly floaty, which wouldn't be a big deal other than that it messes up the timing and tragectory of my grenades. Is this a problem for everyone or is it just me? Is this Bethesda's problem or is there something I can do about it? For reference I'm playing on Xbox, and am up to date with all the latest updates.",
         "         World to me is scary. Expectations from parents and society are making me feel like there is no meaning to life. I am disconnected with everything with zero interests. Maybe it's quarter life crisis maybe depression. I am in a great conflict with stoicism and life in general.",
         "    Deja fucking vu     Didnt play poker for a week. Started to feel pretty good. Then I had the brilliant idea bc I got a little money to deposit $800 and play. Well I lost that in an hour and really didnt have a chance In hell to win. 2 outers, runner runner and a few other hands and there goes that. Im sick again. I dont want to have this feeling ever again. Im so done with this lifestyle and insanity. Its pure fucking evil and has to stop. There is not one thing good that comes from gambling. It makes me miserable, stressed, isolated, broke, scared, depressed, sad, angry. Its a brutal addiction and has really fucked my life up good. Im so fucking pissed at myself that I lost that money when money is so tight and Im barely getting by. -800$ in an hour. Really? Pure fucking stupidity. The whole time Im playing I know Im gonna lose bc its destiny I lose and lose and lose and lose and lose and lose some more so I get in enough pain to stop. When I stop for good thats when I finally win."

    ])