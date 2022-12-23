import torch
import torch.nn as nn
from model.module import *



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        #Sequence Level
        self.seq_emb = Embeddings(config)
        self.seq_norm = LayerNorm(config.hidden_dim)
        self.seq_layers = clones(EncoderLayer(config), config.n_layers)

        #Document Level
        self.doc_emb = PositionalEncoding(config)
        self.doc_norm = LayerNorm(config.hidden_dim)
        self.doc_layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, seq_mask, doc_mask):

        seq_out = self.seq_emb(x)
        for layer in self.seq_layers:
            seq_out = layer(seq_out, seq_mask)
        seq_out = self.seq_norm(seq_out)[:, :, 0]

        doc_out = self.doc_emb(seq_out)
        for layer in self.doc_layers:
            doc_out = layer(doc_out, doc_mask)
        return self.doc_norm(doc_out)



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

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)


    def enc_mask(self, x):
        seq_mask = (x != self.pad_id).unsqueeze(-2).unsqueeze(-2).to(self.device)
        doc_mask = (x[:,:,0] == self.bos_id).unsqueeze(1).unsqueeze(2).to(self.device)
        return seq_mask, doc_mask


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        
        pad_mask = (x != self.pad_id).unsqueeze(-2).unsqueeze(-2)
        sub_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return pad_mask.to(self.device) & sub_mask.to(self.device)


    def forward(self, src, trg):
        seq_mask, doc_mask = self.enc_mask(src)
        d_mask = self.dec_mask(trg)        
        
        memory = self.encoder(src, seq_mask, doc_mask)
        dec_out = self.decoder(trg, memory, doc_mask, d_mask)

        return self.fc_out(dec_out)