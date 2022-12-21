import torch
import torch.nn as nn
from model.module import *



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(DecoderLayer(config), config.n_layers)

    def forward(self, x, memory, e_mask, d_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask)
        return self.norm(x)


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.device = config.device
        self.pad_id = config.pad_id
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)

    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)

    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)

    def forward(self, src, trg):
        e_mask, d_mask = self.pad_mask(src), self.dec_mask(trg)
        memory = self.encoder(src, e_mask)
        dec_out = self.decoder(trg, memory, e_mask, d_mask)
        return self.fc_out(dec_out)