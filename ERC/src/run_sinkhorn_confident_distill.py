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

parser.add_argument('-batch_size', action="store", type=int, default=64)
parser.add_argument('-epochs', action="store", type=int, default=10)
parser.add_argument('-lr', action="store", type=float, default=0.001)
parser.add_argument('-loss_fn', action="store", type=str, default='entropy')
parser.add_argument('-save_model_path', action="store", type=str, default="./")
parser.add_argument('-metric', action="store", type=str, default="micro-f1")
parser.add_argument('-result_path', action="store", type=str, default="./results.txt")
parser.add_argument('-noisy_tranfer_labels', action="store", type=str, default="./results.txt")
parser.add_argument('-lm', action="append", required=True)
parser.add_argument('-pretrained', action="store", default=None)

config = parser.parse_args()
print("\n\n->Configurations are:")
[print(k,": ",v) for (k,v) in vars(config).items()]

#set local variables
num_epochs = config.epochs
BATCH_SIZE= config.batch_size
save_path = config.save_model_path
result_path = config.result_path
metric_type = config.metric
lrate = config.lr
pretrained_model_path = config.pretrained
loss_fn = config.loss_fn

transfer_labels = config.noisy_tranfer_labels
local_models = config.lm

'''
load data
'''
import pickle as pk
import numpy as np

train_load_noisy = pk.load(open(transfer_labels,'rb'))
test_load = [pk.load(open(f_name,'rb')) for f_name in local_models]

def combine_list(lists, ids):
    l_comb = []
    for lis in lists:
        l_comb += lis[ids]
    return l_comb.copy()

data = dict()
#train = transfer data train set
print("\n\n\n >>>LIMITING THE NUMBER OF DATA SAMPLES!!!\n\n\n")

data['train_labels'] = train_load_noisy['train_labels']
data['train_texts'] = train_load_noisy['train_texts']

#val = transfer data val set 
data['val_labels'] = train_load_noisy['val_labels']
data['val_texts'] = train_load_noisy['val_texts']

#test = local data test set
data['test_labels'] = combine_list(test_load, 'test_labels')
data['test_texts'] = combine_list(test_load, 'test_texts')

print("data loading and combination done...")

'''
prepare data
'''
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

MAX_LEN = 200

'''
prepare data for datawise evaluation
'''
train_encodings_comb = tokenizer(data['train_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)
val_encodings_comb = tokenizer(data['val_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)
test_encodings_comb = tokenizer(data['test_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False)

train_dataset_comb = utils.Dataset(train_encodings_comb, data['train_labels'])
val_dataset_comb = utils.Dataset(val_encodings_comb, data['val_labels'])
test_dataset_comb = utils.Dataset(test_encodings_comb, data['test_labels'])

#val_encodings_sep = [tokenizer(test_load[i]['val_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False) for i in range(len(test_load))]
#val_dataset_sep = [utils.Dataset(val_encodings_sep[i], test_load[i]['val_labels']) for i in range(len(test_load))]

test_encodings_sep = [tokenizer(test_load[i]['test_texts'], max_length=MAX_LEN, truncation=True, padding=True, add_special_tokens=True, return_token_type_ids=False) for i in range(len(test_load))]
test_dataset_sep = [utils.Dataset(test_encodings_sep[i], test_load[i]['test_labels']) for i in range(len(test_load))]

'''
define our Transformer model (trainable)
'''
from models import LocalModel, GlobalModel

EMBEDDING_DIM = 128
N_HEAD = 2
N_LAYER = 1
OUTPUT_DIM = len(set(data['test_labels']))

if pretrained_model_path:
    print(f"\nLoading the pretrained model from {pretrained_model_path}...\n")
    model = torch.load(pretrained_model_path)
else:
    print(f"\n--->No loaded pretrained model...\n")
    model = models.GlobalModel(d_model=EMBEDDING_DIM, output_dim=OUTPUT_DIM, nhead=N_HEAD, num_layers=N_LAYER)


print("model intialization done...")

'''
define sinkhorn coordinates and distribution biases
'''
#lab_coordinates = [[-0.4,0.8,0,0,0,0,1], [0.9,0.2,0,0,0,1,0], [0.0,0.0,0,0,1,0,0], [-0.9,-0.4,0,1,0,0,0], [0.4,0.9,1,0,0,0,0]]
lab_coordinates = [[0.0,0,0,0,1], [0.0,0,0,1,0], [0.0,0,1,0,0], [0.0,1,0,0,0], [1.0,0,0,0,0]]
#lab_coordinates = [[-0.4,0.8], [-0.7,0.5], [-0.1,0.8], [0.9,0.2], [0.0,0.0], [-0.9,-0.4], [0.4,0.9]]#, [0.7,0.7], [-0.6,0.4]]
#lab_coordinates = [[0,0,0,0,1,-0.4,0.8], [0,0,0,1,0,0.9,0.2], [0,0,1,0,0,0.0,0.0], [0,1,0,0,0,-0.9,-0.4], [1,0,0,0,0,0.4,0.9]]#, [0.7,0.7], [-0.6,0.4]]
#lab_coordinates = [[-0.4,0.8], [0.9,0.2], [0.0,0.0], [-0.9,-0.4], [0.4,0.9]]#, [0.7,0.7], [-0.6,0.4]]

#mind the underscore in the label coordinate name
lab_coordinates_ = [[-0.4,0.8], [0.9,0.2], [0.0,0.0], [-0.9,-0.4], [0.4,0.9]]#, [0.7,0.7], [-0.6,0.4]]
lab_coordinate_dict = {i:lab_coordinates_[i] for i in range(OUTPUT_DIM)}

#in order meld, iemocap, dyda
biases = [[0.89919, 0.00227, 0.09622, 0.00236, 1e-05],
          [0.06909, 0.69452, 0.2364, 1e-05, 3e-05],
          [1e-05, 2e-05, 0.99994, 1e-05, 7e-05],
          [1e-05, 0.00013, 0.99989, 1e-05, 1e-05],
          [1e-05, 0.00016, 0.99984, 1e-05, 3e-05]
          ]

'''
Evaluate function
'''
f1_metric = datasets.load_metric("f1")
acc_metric = datasets.load_metric("accuracy")

def reg_score(all_targets, all_outputs):
    #consider x coordinate for list
    lab_list = [int(k) for k in set(all_targets.tolist())]
    labwise_score = {k:[] for k in lab_list}

    src_supp_x = torch.tensor([l[0] for l in lab_coordinates_])
    src_supp_y = torch.tensor([l[1] for l in lab_coordinates_])

    src_supp_x = src_supp_x.repeat(all_targets.shape[0],1).cuda()
    src_supp_y = src_supp_y.repeat(all_targets.shape[0],1).cuda()

    all_targets_ = all_targets.tolist()
    all_targets_ = [lab_coordinate_dict[l] for l in all_targets_]

    all_targets_x = torch.tensor([l[0] for l in all_targets_]).cuda()
    all_targets_y = torch.tensor([l[1] for l in all_targets_]).cuda()

    all_outputs = torch.nn.Softmax()(all_outputs)

    print((all_outputs * src_supp_x).sum(dim=1)[:5], all_targets_x[:5])
    print((all_outputs * src_supp_y).sum(dim=1)[:5], all_targets_y[:5])

    score_x = (all_outputs * src_supp_x).sum(dim=1) - all_targets_x
    score_y = (all_outputs * src_supp_y).sum(dim=1) - all_targets_y

    score = torch.sqrt(score_x**2 + score_y**2).detach().cpu() #abs is important
    #score = torch.mean(score)

    #load scores of lab-specific samples
    for t in range(len(all_targets)):
        labwise_score[int(all_targets[t])].append(abs(score[t]))

    #lab-specific mean score
    labwise_mean = {k:sum(labwise_score[k])/len(labwise_score[k]) for k in lab_list}

    #kind of F1-score over all labs
    mean_score = sum(labwise_mean.values())/len(lab_list)

    return mean_score


'''
Evaluate function
'''
#if len(data['train_texts'])//BATCH_SIZE < 50:
#    BATCH_SIZE = len(data['train_texts'])//50

print(f"BATCH_SIZE: {BATCH_SIZE}")

def evaluate(val_dat, dat_type=''):
    #
    model.eval()
    #
    val_loader = DataLoader(val_dat, batch_size=BATCH_SIZE, shuffle=False)
    #
    all_preds = []
    all_outputs = []
    all_targets = []
    #
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            print(f"Batch: [ {i}|{len(val_loader)} ]", end='\r')
            #
            input_ids = [batch['input_ids'].to(device),batch['attention_mask'].to(device)]
            #
            labels = batch['labels'].to(device)
            #
            outputs = model(input_ids)
            #
            all_outputs += outputs.tolist()
            #
            all_preds += outputs.argmax(dim=1).tolist()
            #
            all_targets += labels.tolist()

        all_preds = torch.tensor(all_preds).cuda()
        all_outputs = torch.tensor(all_outputs).cuda()
        all_targets = torch.tensor(all_targets).cuda()

        if metric_type == 'f1_macro' and 'test' in dat_type:
            score = f1_metric.compute(references=all_targets.tolist(), predictions=all_preds.tolist(), average='macro')['f1']
            loss = reg_score(all_targets, all_outputs)
            report = classification_report(all_targets.cpu().numpy(), all_preds.cpu().numpy())
            print(f"\n classification report\n", report)
            print("\nReg loss:", loss)

        elif metric_type == 'accuracy' and 'test' in dat_type:
            score = acc_metric.compute(references=all_targets.tolist(), predictions=all_preds.tolist())['accuracy']
            loss = reg_score(all_targets, all_outputs)
            report = classification_report(all_targets.cpu().numpy(), all_preds.cpu().numpy())
            print(f"\n classification report\n", report)
            print("\nReg loss:", loss)
        
        elif 'test' in dat_type:
            print('\nmetric unrecognised, using accuracy!')
            score = acc_metric.compute(references=all_targets.tolist(), predictions=all_preds.tolist())['accuracy']
            loss = reg_score(all_targets, all_outputs)
            report = classification_report(all_targets.cpu().numpy(), all_preds.cpu().numpy())
            print(f"\n classification report\n", report)
            print('\nReg loss:', loss)
        
        else:
            loss = loss_func(all_outputs, all_targets, lab_coordinates, biases, device)
            report = None
            print(f"\n {loss_fn} loss:", loss)

        #if metric_type == 'f1_macro':
        #    score = f1_metric.compute(references=all_targets, predictions=all_preds, average='macro')['f1']
        #elif metric_type == 'accuracy':
        #    score = acc_metric.compute(references=all_targets, predictions=all_preds)['accuracy']
        #else:
        #    print('\nmetric unrecognised, using accuracy!')
        #    score = acc_metric.compute(references=all_targets, predictions=all_preds)['accuracy']
        #

        print(f"\n\n[{dat_type}]")
        #print(f"{metric_type}_score is {score}")
        #
        #full report
        #report = classification_report(all_targets, all_preds)
        #print(f"\n classification report\n", report)
        if 'test' in dat_type:
            return loss, report
        else:
            return loss, report

'''
Train
'''
model.to(device)

#optimizer
optimizer = optim.AdamW(model.parameters(), lr=lrate)

#loss function
if loss_fn=='entropy':
    print("\n[entropy loss!]\n")
    from utils import entropy_loss as loss_func
elif loss_fn=='ot':
    print("\n[sinkhorn loss!]\n")
    from utils import sinkhorn_loss as loss_func

#few important variables

best_val_loss, best_val_score = evaluate(val_dataset_comb, "validation")

best_test_score_combined = evaluate(test_dataset_comb, "test")
best_test_score_sep = [evaluate(test_dataset_sep[i], f"test-{local_models[i]}") for i in range(len(local_models))]
best_test_score_mean = sum([v[0] for v in best_test_score_sep])/len(local_models)

#initialise train loader
print("Changed train set to val set!!!")
train_loader = DataLoader(train_dataset_comb, batch_size=BATCH_SIZE, shuffle=True)
best_epoch = 0

#training epochs
import time

f_times = []
l_times = []
b_times = []
g_times = []
it_times = []

for epoch in range(num_epochs):
    print(f"-----> Epoch: {epoch}")
    model.train()
    for i, batch in enumerate(train_loader):

        #flush gradients
        optimizer.zero_grad()
        
        #
        input_ids = [batch['input_ids'].to(device),batch['attention_mask'].to(device)]
        
        labels = batch['labels'].to(device)

        #start time
        s_time = time.time()

        #feed inputs to the model
        output = model(input_ids)
        
        #time for a forward pass
        f_time = time.time() - s_time

        #calculate the loss
        loss = loss_func(output, labels, lab_coordinates, biases, device)
        
        #time for loss calculation
        l_time = time.time() - (f_time+s_time)

        #compute gradients
        loss.backward()
        
        #time for a backward pass
        b_time = time.time() - (f_time+s_time+l_time)

        #gradient step
        optimizer.step()
        
        #time for a gradient step
        g_time = time.time() - (f_time+s_time+l_time+b_time)

        #time for an iteration
        it_time = f_time+l_time+b_time+g_time

        #show output
        print(f"Batch: {i+1} | {len(train_loader)} | loss:{loss}", end='\r')

        f_times.append(f_time)
        l_times.append(l_time)
        b_times.append(b_time)
        g_times.append(g_time)
        it_times.append(it_time)

    print("Avg times: foward={} | {} loss={} | backward={} | gradStep={} | epoch={}".format(
                                                                                sum(f_times)/len(f_times),
                                                                                loss_fn,
                                                                                sum(l_times)/len(l_times),
                                                                                sum(b_times)/len(b_times),
                                                                                sum(g_times)/len(g_times),
                                                                                sum(it_times)
                                                                                ))


    val_loss, val_score = evaluate(val_dataset_comb, "validation")

    test_score_combined = evaluate(test_dataset_comb, "test")
    test_score_sep = [evaluate(test_dataset_sep[i], f"test-{local_models[i]}") for i in range(len(local_models))]
    test_score_mean = sum([v[0] for v in test_score_sep])/len(local_models)

    #save if it's the best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_score = val_score

        best_test_score_combined = test_score_combined
        best_test_score_sep = test_score_sep
        best_test_score_mean = test_score_mean

        best_epoch = epoch
        print('Saving best model...\n\n')
        torch.save(model, f"{save_path}/best_model_global_finetune_{loss_fn}_best_model.pt")

print('Best score is...')
print(f"best epoch: {best_epoch} | Best Valid Loss: {best_val_loss:.3f}% | Test Loss: {best_test_score_mean:.3f}")
print(f"saved in {result_path}")

import os
import datetime

mode = 'a' if os.path.exists(result_path) else 'w'
with open(result_path, mode) as f:
    f.write(f"---> student | transfer | {loss_fn} | euclidean-d | p=1 | lab coord: {lab_coordinates} |date-time: {datetime.datetime.now()}\n")
    f.write(f"class proportion | train: {[round(data['test_labels'].count(i)/len(data['test_texts']),3) for i in range(OUTPUT_DIM)]}\n")
    f.write(f'Epoch: {best_epoch} | avg over datasets | Best valid loss: {best_val_loss} | Test loss: {best_test_score_mean}\n')
    f.write(f'Epoch: {best_epoch} | avg over data points | Best valid loss: {best_val_loss} | Test loss: {best_test_score_combined[1]}\n')
    f.write(f"Datawise test fscore at best val fscore overall: \n")
    for i,f_name in enumerate(local_models):
        f.write(f"--> {f_name}\n: {best_test_score_sep[i][1]}\n")
        f.write(f"--> Reg score: {best_test_score_sep[i][0]}\n\n\n")
    #
    f.write(f"\n\n\n")
