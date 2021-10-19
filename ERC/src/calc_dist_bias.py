#link: https://huggingface.co/transformers/custom_datasets.html
#https://arxiv.org/pdf/1908.08962.pdf

import torch
import utils
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast
from models import LocalModel, GlobalModel

import pickle as pk
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import argparse
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-batch_size', action="store", type=int, default=64)
parser.add_argument('-lm', action="append", required=True)
parser.add_argument('-gm', action="store", type=str, required=True)

config = parser.parse_args()
print("\n\n->Configurations are:")
[print(k,": ",v) for (k,v) in vars(config).items()]

#data file name
local_models = config.lm #['best_model_iemocap.pt', 'best_model_meld_e.pt']
global_model = config.gm #'best_model_dyda_e.pt'
batch_size = config.batch_size


print("\nEstimating class distribution from models:",*(local_models+[global_model]), sep='\n\t')
#load local models      
local_models = [torch.load(f"../saved_models/{m_name}").to(device).eval() for m_name in local_models]
#load global model
global_model = [torch.load(f"../saved_models/{global_model}").to(device).eval()]


'''
prepare random data
'''
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

from random import choices

MAX_LEN = 512
vocab = list(tokenizer.get_vocab().keys())
num_samples = 100000
random_texts = [" ".join(choices(vocab,k=MAX_LEN)) for i in range(num_samples)]
random_labels = [0 for i in range(num_samples)]

train_encodings = tokenizer(random_texts, max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)
train_dataset = train_dataset = utils.Dataset(train_encodings, random_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
teacher_labels = []
for i, batch in enumerate(train_loader):
    with torch.no_grad():
        print(f"Batch: [{i}|{len(train_loader)}]", end='\r')
        #
        input_ids = [batch['input_ids'].to(device), batch['attention_mask'].to(device)]
        #
        labels = batch['labels'].to(device)
        #
        outputs = [torch.nn.Softmax(dim=1)(model(text=input_ids)) for model in local_models+global_model]
        #
        teacher_labels.append([np.argmax(output.cpu().numpy(), axis=1) for output in outputs])

#adding one sample of each class to have non zero probability 
teacher_labels = [np.concatenate([teacher_labels[i][j] for i in range(len(teacher_labels))]) for j in range(len(teacher_labels[0]))]
teacher_labels = [np.concatenate([lab, np.array(range(global_model[0].fc_final.out_features))]) for lab in teacher_labels]

model_names = config.lm + [config.gm]
count_occ = [(model_names[i], list(np.unique(teacher_labels[i], return_counts=True)[1]/num_samples)) for i in range(len(model_names))]

print(*count_occ,sep='\n')

print("Note: write the above in CONFIG file under domain_distr")
