import torch
import models
import utils

from transformers import BertTokenizerFast

from torch.utils.data import DataLoader
import torch.optim as optim

import pickle as pk
import numpy as np

#performance metrics
from sklearn.metrics import classification_report
import datasets

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

'''
    user-specifications
'''
import argparse
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-dataset', action="store", type=str)
parser.add_argument('-batch_size', action="store", type=int, default=64)
parser.add_argument('-epochs', action="store", type=int, default=10)
parser.add_argument('-lr', action="store", type=float, default=0.001)
parser.add_argument('-max_text_len', action="store", type=int, default=512)
parser.add_argument('-loss_fn', action="store", type=str, default='entropy')
parser.add_argument('-save_model_path', action="store", type=str, default="./")
parser.add_argument('-metric', action="store", type=str, default="micro-f1")
parser.add_argument('-result_path', action="store", type=str, default="./results.txt")
parser.add_argument('-model_type', action="store", type=str, default="local")
#parser.add_argument('-conversation', action="store_true", default=False)

config = parser.parse_args()
print("\n\n->Configurations are:")
[print(k,": ",v) for (k,v) in vars(config).items()]

#set local variables
MAX_LEN = config.max_text_len
num_epochs = config.epochs
fname = config.dataset
BATCH_SIZE= config.batch_size
save_path = config.save_model_path
result_path = config.result_path
metric_type = config.metric
lrate = config.lr
model_type = config.model_type
loss_fn = config.loss_fn


'''
    load data and stats
'''
print(f"Dataset [{fname}]...\n")
with open(fname,'rb') as f_open:
    data = pk.load(open(fname,'rb'))

print(f"total number of samples {len(data['train_labels'])}")

print("Class proportion")
class_prop = {}
for lab in set(data['train_labels']+data['val_labels']):
    class_prop[lab] = round(data['train_labels'].count(lab)/len(data['train_texts']),3)

    print(f"Label: {lab} has fraction {class_prop[lab]}")


'''
tokenize and torchise the data
'''

#tokenize the data
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(data['train_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)
val_encodings = tokenizer(data['val_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)
test_encodings = tokenizer(data['test_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)

#prepare dataset for torch
train_dataset = utils.Dataset(train_encodings, data['train_labels'])
val_dataset = utils.Dataset(val_encodings, data['val_labels'])
test_dataset = utils.Dataset(test_encodings, data['test_labels'])


'''
define our Transformer model (trainable)
'''
EMBEDDING_DIM = 128
N_HEAD = 2
N_LAYER = 1
HIDDEN_DIM = EMBEDDING_DIM
OUTPUT_DIM = len(set(data['train_labels'] + data['val_labels']))

if model_type == 'local':
    print("\n\nrunning as local model!\n\n")
    model = models.LocalModel(d_model=EMBEDDING_DIM, output_dim=OUTPUT_DIM, nhead=N_HEAD, num_layers=N_LAYER)
    #
elif model_type == 'global':
    print("\n\nrunning as global model!\n\n")
    model = models.GlobalModel(d_model=EMBEDDING_DIM, output_dim=OUTPUT_DIM, nhead=N_HEAD, num_layers=N_LAYER)
    #
else:
    print(f"\n\nWarning! Model type {model_type} ","is not from {local, global}")
    print("running as local model!\n\n")


'''
Evaluate function
'''
f1_metric = datasets.load_metric("f1")
acc_metric = datasets.load_metric("accuracy")

def evaluate(val_dat, dat_type=''):
    #
    model.eval()
    #
    val_loader = DataLoader(val_dat, batch_size=BATCH_SIZE, shuffle=False)
    #
    all_preds = []
    all_targets = []
    #
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            print(f"Batch: [ {i}|{len(val_loader)} ]", end='\r')
            #
            input_ids = [batch['input_ids'].to(device),batch['attention_mask'].to(device)]
            #
            labels = batch['labels'].to(device)
            #
            outputs = model(input_ids)
            #
            all_preds += outputs.argmax(dim=1).detach().cpu().numpy().tolist()
            #
            all_targets += labels.detach().cpu().numpy().tolist()

    if metric_type == 'f1_macro':
        score = f1_metric.compute(references=all_targets, predictions=all_preds, average='macro')['f1']
    elif metric_type == 'accuracy':
        score = acc_metric.compute(references=all_targets, predictions=all_preds)['accuracy']
    else:
        print('\nmetric unrecognised, using accuracy!')
        score = acc_metric.compute(references=all_targets, predictions=all_preds)['accuracy']
    #
    print(f"\n\n[{dat_type}]")
    print(f"{metric_type}_score is {score}")
    #
    #full report
    report = classification_report(all_targets, all_preds)
    print(f"\n classification report\n", report)
    #
    return score, report


'''
Train
'''
model.to(device)

#optimizer
optimizer = optim.AdamW(model.parameters(), lr=lrate)

#loss function
if loss_fn=='entropy':
    print("\n[entropy loss!]\n")
    from utils import entropy_loss_loc as loss_func
elif loss_fn=='ot':
    print("\n[sinkhorn loss!]\n")
    from utils import sinkhorn_loss_loc as loss_func

#few important variables
best_val = 0.0
best_val_rep = ''
test_at_best_val = 0.0
best_test_rep = ''

best_epoch = 0

#define sinkhorn coordinates
lab_coordinates = [[-0.4,0.8], [-0.7,0.5], [-0.1,0.8], [0.9,0.2], [0.0,0.0], [-0.9,-0.4], [0.4,0.9]]#, [0.7,0.7], [-0.6,0.4]]

#initialise train loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#training epochs
for epoch in range(num_epochs):
    print(f"-----> Epoch: {epoch}")
    model.train()
    for i, batch in enumerate(train_loader):
        #
        optimizer.zero_grad()
        #
        input_ids = [batch['input_ids'].to(device),batch['attention_mask'].to(device)]
        #
        labels = batch['labels'].to(device)
        #
        outputs = model(input_ids)
        #
        loss = loss_func(outputs, labels, lab_coordinates, device)#entropy_loss(outputs, labels)
        #
        loss.backward()
        #
        optimizer.step()
        #
        print(f"Batch: {i+1} | {len(train_loader)} | loss:{loss}", end='\r')


    val_score, val_class_rep = evaluate(val_dataset, "validation")
    test_score, test_class_rep = evaluate(test_dataset, "test")
    #
    #save if it's the best model
    if val_score > best_val:
        best_val = val_score
        best_val_rep = val_class_rep
        test_at_best_val = test_score
        best_test_rep = test_class_rep
        best_epoch = epoch
        print('Saving best model...\n\n')
        torch.save(model, f"{save_path}/best_model_{loss_fn}_{fname.split('/')[-1].replace('pkl','pt')}")


'''
save results
'''
print('Best score is...')
print(f"best epoch: {best_epoch} | Best Valid Acc: {best_val:.3f}% | Test Acc: {test_at_best_val:.3f}")
print(f"saved in {result_path}")

import os
import datetime

mode = 'a' if os.path.exists(result_path) else 'w'
with open(result_path, mode) as f:
    f.write(f"---> local model | {loss_fn} | data: {fname} | date-time: {datetime.datetime.now()}\n")
    f.write(f"class proportion | train+valid: {class_prop}\n")
    f.write(f"number of samples | train: {len(data['train_texts'])} | val: {len(data['val_labels'])} | test: {len(data['test_labels'])}\n")
    f.write(f"full test report \n {best_test_rep}\n\n")
    f.write(f'best epoch: {best_epoch} | Best Valid Acc: {best_val:.3f}% | Test Acc: {test_at_best_val:.3f}%\n\n\n')
