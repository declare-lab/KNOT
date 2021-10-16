'''
utility functions
'''
import torch
import copy
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss


#assign ids to utterances
def get_conv_ids(encodings, tokenizer):
    print('\nEncoding utterance ids...')
    conv_ids = []
    sep_id = tokenizer.sep_token_id
    encods = encodings['input_ids']
    for utt in encods:
        id_track = 1.0
        conv_ids.append([])
        for token in utt:
            conv_ids[-1] = conv_ids[-1] + [id_track]
            if token == sep_id:
                id_track += 1.0
    return conv_ids


'''
define Dataset parser
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    #
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    #
    def __len__(self):
        return len(self.labels)


'''
standard entropy loss
'''
cross_ent_loss = torch.nn.CrossEntropyLoss()
def entropy_loss_loc(output, target, lab_coordinates, device):

    #OUTPUT_DIM = output.shape[-1]

    #soft_outputs = torch.nn.Softmax()(output)

    #target_probs = torch.nn.functional.one_hot(target, num_classes=OUTPUT_DIM)
    #target_probs = torch.nn.Softmax()(target_probs+1e-5) #adding non-zero prob over the support

    #L_conf = (0.5*((soft_outputs * (soft_outputs / target_probs).log()) + (target_probs * (target_probs / soft_outputs).log())))

    #L_conf = torch.mean( L_conf )

    L_conf = cross_ent_loss(output, target)

    return L_conf


'''
standard optimal transport loss
'''
def sinkhorn_loss_loc(output, target, lab_coordinates, device):
    src_supp = torch.tensor(lab_coordinates, requires_grad=True).to(device)
    trg_supp = torch.tensor(lab_coordinates, requires_grad=True).to(device)

    src_supp = src_supp.repeat(output.shape[0],1,1).cuda()
    trg_supp = trg_supp.repeat(target.shape[0],1,1).cuda()

    OUTPUT_DIM = output.shape[-1]

    soft_output = torch.nn.Softmax()(output)
    loss_wass = SamplesLoss(loss="sinkhorn", debias=False, reach=None, p=1, blur=0.001, backend="tensorized").to(device)

    #for evaluation (one-hot and making non-zero probability of each class)
    
    target_probs = torch.nn.functional.one_hot(target, num_classes=OUTPUT_DIM)
    
    target_probs = target_probs + 1e-5
    
    target_probs = target_probs/(target_probs.sum(dim=1).unsqueeze(-1))
    
    D = loss_wass(soft_output, src_supp, target_probs, trg_supp)

    L = torch.mean(D) + cross_ent_loss(output, target)

    return L


'''
distillation entropy loss
'''
def entropy_loss(output, target, lab_coordinates, biases, device):

    OUTPUT_DIM = output.shape[-1]
    soft_outputs = torch.nn.Softmax()(output)

    unf = [torch.tensor(dist) for dist in biases]
    unf = [unf[i].repeat(output.shape[0],1).to(device) for i in range(len(unf))]

    L_conf = 0.0
    d_norm = 0.0

    dat_dist = [0.16, 0.23, 0.14, 0.47]

    for j in range(target[0].shape[0]//OUTPUT_DIM): 

        lab = target[:, j*OUTPUT_DIM: (j+1)*OUTPUT_DIM]

        #d_bias = 1.0
        #d_bias = torch.nn.PairwiseDistance(p=2)(lab, unf[j]).view(-1,1)
        d_bias = dat_dist[j]

        L_conf += d_bias*(0.5*((soft_outputs * (soft_outputs / lab).log()) + (lab * (lab / soft_outputs).log())))
        #L_conf += -d_bias*(lab * torch.log2(soft_outputs)) #cross-entropy
        #d_norm += d_bias
        
    #
    #L_conf = torch.mean( (1/d_norm) * L_conf )
    L_conf = torch.mean( L_conf )

    return L_conf


'''
distillation optimal transport loss
'''
def sinkhorn_loss(output, target, lab_coordinates, biases, device):

    src_supp = torch.tensor(lab_coordinates, requires_grad=True).to(device)
    trg_supp = torch.tensor(lab_coordinates, requires_grad=True).to(device)

    src_supp = src_supp.repeat(output.shape[0],1,1).cuda()
    trg_supp = trg_supp.repeat(target.shape[0],1,1).cuda()

    OUTPUT_DIM = output.shape[-1]

    soft_output = torch.nn.Softmax()(output)
    loss_wass = SamplesLoss(loss="sinkhorn", debias=False, reach=None, p=1, blur=0.001, backend="tensorized").to(device)

    '''
    if target[0].dim() == 0:
        #for evaluation

        target_probs = torch.nn.functional.one_hot(target, num_classes=OUTPUT_DIM)
        target_probs = target_probs + 1e-5
        target_probs = target_probs/(target_probs.sum(dim=1).unsqueeze(-1))
        D = loss_wass(soft_output, src_supp, target_probs, trg_supp)
        L_conf = torch.mean(D)
    
    else:
    #for training
    '''

    target_probs = target
    unf = [torch.tensor(dist) for dist in biases] #found from random inputs
    unf = [unf[i].repeat(output.shape[0],1).to(device) for i in range(len(unf))]

    L_conf = 0.0
    d_norm = 0.0
    dat_dist = [0.16, 0.23, 0.14, 0.47]

    for j in range((target[0].shape[0])//OUTPUT_DIM):

        lab = target[:, j*OUTPUT_DIM: (j+1)*OUTPUT_DIM]

        #d_bias = 1.0
        #d_bias = torch.nn.PairwiseDistance(p=2)(lab, unf[j])
        d_bias = dat_dist[j]
        
        D_wass = loss_wass(soft_output, src_supp, lab, trg_supp)

        L_conf += d_bias*D_wass
        #d_norm += d_bias

    #L_conf = torch.mean( (1/d_norm) * L_conf )
    L_conf = torch.mean( L_conf )

    return L_conf
