# Federated Distillation of Natural Language Understanding with Confident Sinkhorns

This repository provides an alternative method for ensembled distillation of local models to a global model. The local models can be trained via entropy or optimal transport (OT) loss. We train local (on-device) models using cross-entropy loss due to the higher computational complexity of OT. The global model is pretrained on global dataset which is relatively bigger than local datasets.

## How to run?

For the Sentiment task, in the Sentiment directory
  
    Within the dataset directory:
    - Follow the folder-specific readme to download the datasets and preprocess.

    Within the src directory:
    - To pretrain local models: run scripts for local models mentioned in bash.sh file under the comment line #train local models.
    
    - To pretrain global model: run the script for global model mentioned in bash.sh file under the comment line #pretrain global model.
    
    - To create noisy labels: run the script mentioned in bash.sh file under the comment line #create noisy labels from local models on transfer set.
    
    - To find pretrained local and global model bias: run the script mentioned in bash.sh file under the comment line #distribution bias.
    
    - To distil knowledge from pretrained local and global model: run the script mentioned in bash.sh file under the comment line #distill knowledge.


[Read the paper](https://arxiv.org/pdf/2110.02432.pdf)
