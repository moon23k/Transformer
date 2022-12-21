import torch
import torch.nn as nn
from model.module import *



class SequenceEncoder(nn.Module):
    def __init__(self, config):
        super(SequenceEncoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)[:, :, 0]


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super(ContextEncoder, self).__init__()
        self.pos_enc = PositionalEncoding(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, mask):
        x = self.pos_enc(x)
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


class HierModel(nn.Module):
    def __init__(self, config):
        super(HierModel, self).__init__()
        self.device = config.device

        self.pad_id = config.pad_id
        self.bos_id = config.bos_id

        self.sequence_encoder = SequenceEncoder(config)
        self.context_encoder = ContextEncoder(config)
        
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)


    def enc_mask(self, x):
        tok_mask = (x != self.pad_id).unsqueeze(-2).unsqueeze(-2).to(self.device)
        seq_mask = (x[:,:,0] == self.bos_id).unsqueeze(1).unsqueeze(2).to(self.device)
        return tok_mask, seq_mask


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        
        pad_mask = (x != self.pad_id).unsqueeze(-2).unsqueeze(-2)
        sub_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return pad_mask.to(self.device) & sub_mask.to(self.device)


    def forward(self, src, trg):
        tok_mask, seq_mask = self.enc_mask(src)
        d_mask = self.dec_mask(trg)        
        
        sequence_memory = self.sequence_encoder(src, tok_mask)
        context_memory = self.context_encoder(sequence_memory, seq_mask)
        dec_out = self.decoder(trg, context_memory, seq_mask, d_mask)
        return self.fc_out(dec_out)