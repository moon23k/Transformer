import torch
import torch.nn as nn
from collections import namedtuple
from .common import (
    clones, 
    Embeddings, 
    SublayerConnection, 
    PositionwiseFeedForward
)




class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, 
            config.n_heads, 
            batch_first=True
        )
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, mask):
        x = self.sublayer[0](
            x, lambda x: self.attn(
                x, x, x, key_padding_mask=mask, need_weights=False
                )[0]
            )
        return self.sublayer[1](x, self.pff)




class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, 
            config.n_heads, 
            batch_first=True
        )

        self.enc_attn = nn.MultiheadAttention(
            config.hidden_dim, 
            config.n_heads, 
            batch_first=True
        )

        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)


    def forward(self, x, m, e_mask, d_mask):
        x = self.sublayer[0](
            x, lambda x: self.self_attn(
                x, x, x, attn_mask=d_mask, need_weights=False
                )[0]
            )

        x = self.sublayer[1](
            x, lambda x: self.enc_attn(
                x, m, m, key_padding_mask=e_mask, need_weights=False
                )[0]
            )

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




class HybridModel(nn.Module):
    def __init__(self, config):
        super(HybridModel, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

    
    def pad_mask(self, x):
        return x == self.pad_id


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, x, y):
        y, label = self.shift_y(y)

        #Masking
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)
        
        #Actual Processing
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