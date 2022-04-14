from transformers import BertForSequenceClassification, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
import pdb
import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained("models/checkpoint-15546").to(device)
model.eval()

softmax = nn.Softmax(dim=-1)

text = "This has helped me a lot today, Thank you. I was diagnosed with anxiety and depression about 15 years ago. Today isn't a good day for me, one of the many thoughts in my head is that I should 'suck it up' and 'other people have it worse'. I know it won't last and then I'll feel silly/stupid for my mood. Want to give a shout out to my partner and family. They have gone through it all with me. Everyone deserves to be heard"

token_dict = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=400, \
                       return_token_type_ids=False, return_attention_mask=False)
token_ids = token_dict['input_ids'].to(device)
with torch.no_grad():
    output = model(token_ids)
    output = softmax(output.logits)
    labels = torch.argmax(output, axis=-1)

print(labels.tolist()[0])
