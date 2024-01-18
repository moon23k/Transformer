import torch
import torch.nn as nn
from collections import namedtuple
from .common import (
    clones, 
    Embeddings, 
    SublayerConnection,
    MultiHeadAttention, 
    PositionwiseFeedForward
)




class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pff)




class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.src_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, m, e_mask, d_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, d_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, e_mask))
        return self.sublayer[2](x, self.pff)




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()        

        self.embeddings = Embeddings(config)
        layer = EncoderLayer(config)
        self.layers = clones(layer, config.n_layers)

    def forward(self, x, e_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, e_mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.embeddings = Embeddings(config)        
        layer = DecoderLayer(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, m, e_mask, d_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, m, e_mask, d_mask)
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