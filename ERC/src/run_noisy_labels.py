import torch
import utils
from models import LocalModel, GlobalModel

import numpy as np
import pickle as pk

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import argparse
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-tranfer_set', action="store", type=str)
parser.add_argument('-batch_size', action="store", type=int, default=64)
parser.add_argument('-save_model_path', action="store", type=str, default="./")
parser.add_argument('-lm', action="append", required=True)
parser.add_argument('-gm', action="store", type=str, required=True)

config = parser.parse_args()
print("\n\n->Configurations are:")
[print(k,": ",v) for (k,v) in vars(config).items()]

#data file name
local_models = config.lm #['best_model_iemocap.pt', 'best_model_meld_e.pt']
global_model = config.gm #'best_model_dyda_e.pt'
transfer_set = config.tranfer_set
batch_size = config.batch_size
save_path = config.save_model_path


print("\nInfering from models:",*(local_models+[global_model]), sep='\n\t')
#load local models      
local_models = [torch.load(f"{m_name}").to(device).eval() for m_name in local_models]
#load global model
global_model = [torch.load(f"{global_model}").to(device).eval()]

print(f"\n\nworking on the data-[{transfer_set}]")

#load transfer set
with open(f"../train_test_split/{transfer_set}",'rb') as file:
    data = pk.load(file)

#define tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#inference
teacher_labels_np = {}

for dat_type in ["train", "val"]:
    MAX_LEN = 512
    train_encodings = tokenizer(data[dat_type+"_texts"], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)

    train_dataset = utils.Dataset(train_encodings, data[dat_type+"_labels"])

    teacher_labels = []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
            teacher_labels.append([output.cpu().numpy() for output in outputs])

    teacher_labels_np[dat_type] = np.concatenate([np.concatenate(teacher_labels[i], axis=1) for i in range(len(teacher_labels))])

with open(f"{save_path+'/'+transfer_set.split('/')[-1]}", 'wb') as file:
    pk.dump({'train_labels':teacher_labels_np["train"],
            'train_texts':data['train_texts'], 
            'val_labels':teacher_labels_np["val"],
            'val_texts':data['val_texts'],
            'test_labels':data['test_labels'], 
            'test_texts':data['test_texts'],
            }
            , file)

print(f"dumped in dir: {save_path+'/'+transfer_set.split('/')[-1]}")
