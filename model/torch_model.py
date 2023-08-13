import torch
import torch.nn as nn
from collections import namedtuple
from model.common import clones, shift_trg, Embeddings




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, memory_key_padding_mask, tgt_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_mask=tgt_mask,
            )
        return x



class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')

    
    def pad_mask(self, x):
        return x == self.pad_id
    

    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)

        
    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        #Masking
        e_mask = self.pad_mask(src)
        d_mask = self.dec_mask(trg)
        
        #Actual Processing
        memory = self.encoder(src, src_key_padding_mask=e_mask)
        
        dec_out = self.decoder(
            trg, memory, 
            memory_key_padding_mask=e_mask,
            tgt_mask=d_mask
        )
        
        logit = self.generator(dec_out)
        

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out
