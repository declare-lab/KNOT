from transformers import BertModel
import torch
import torch.nn as nn


'''
define the Transformer-based model
'''
class LocalModel(nn.Module):
    def __init__(self, d_model, output_dim, nhead, num_layers, layer_norm_eps: float = 1e-5):
        super().__init__()
        #
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2', output_hidden_states=True)
        #
        self.TransformerEncoder = nn.TransformerEncoder(
                                                        nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
                                                        num_layers=num_layers
                                                        )
        #
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        #
        self.fc_final = nn.Linear(d_model, output_dim)
        #
    def forward(self, text):
        #
        inp = self.bert(text[0], attention_mask=text[1], token_type_ids=None)
        #
        add_norm = self.norm(inp[0]+inp[2][0]) #residual: output of last layer + bert layer 0 embedding (non-contextualized)
        #
        hidden = self.TransformerEncoder(add_norm.transpose(0,1), src_key_padding_mask= ~text[1].type(torch.bool))
        #
        out = self.fc_final(hidden[0,:,:])
        #
        return out


#In the paper, global and local architectures are kept same
class GlobalModel(nn.Module):
    def __init__(self, d_model, output_dim, nhead, num_layers, layer_norm_eps: float = 1e-5):
        super().__init__()
        #
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2', output_hidden_states=True)
        #
        self.TransformerEncoder = nn.TransformerEncoder(
                                                        nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
                                                        num_layers=num_layers
                                                        )
        #
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        #
        self.fc_final = nn.Linear(d_model, output_dim)
        #
    def forward(self, text):
        #
        inp = self.bert(text[0], attention_mask=text[1], token_type_ids=None)
        #
        add_norm = self.norm(inp[0]+inp[2][0]) #residual: output of last layer + bert layer 0 embedding (non-contextualized)
        #
        hidden = self.TransformerEncoder(add_norm.transpose(0,1), src_key_padding_mask= ~text[1].type(torch.bool))
        #
        out = self.fc_final(hidden[0,:,:])
        #
        return out
