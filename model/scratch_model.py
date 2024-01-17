import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .common import (
    clones, 
    Embeddings, 
    MultiHeadAttention, 
    PositionwiseFeedForward
)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        hidden_dim = config.hidden_dim
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)


    def forward(self, x, e_mask):
        _x = self.self_attn(x, x, x, e_mask)
        x = self.self_attn_norm(x + self.dropout1(_x))
        _x = self.pff(x)
        x = self.pff_norm(x + self.dropout2(_x))

        return x



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.enc_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        hidden_dim = config.hidden_dim
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_norm = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)
        self.dropout3 = nn.Dropout(config.dropout_ratio)


    def forward(self, x, m, e_mask, d_mask):

        _x = self.self_attn(x, x, x, d_mask)
        x = self.self_attn_norm(x + self.dropout1(_x))

        _x = self.enc_attn(x, m, m, e_mask)
        x = self.enc_attn_norm(x + self.dropout2(_x))

        _x = self.pff(x)
        x = self.pff_norm(x + self.dropout3(_x))

        return x



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device
        self.emb = Embeddings(config)

        layer = EncoderLayer(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, e_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, e_mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device
        self.emb = Embeddings(config)

        layer = DecoderLayer(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask, d_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask)
        return x
     


class ScratchModel(nn.Module):
    def __init__(self, config):
        super(ScratchModel, self).__init__()

        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        sz = x.size(1)
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((sz, sz), device=self.device)).bool()
        return pad_mask & sub_mask


    @staticmethod
    def shift_y(x):
        return x[:, :-1], x[:, 1:]


    def forward(self, x, y):
        y, label = self.shift_y(y)

        e_mask = self.pad_mask(x) 
        d_mask = self.dec_mask(y)

        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out