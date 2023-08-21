import torch
import torch.nn as nn
from .common import clones, Embeddings, ModelBase




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


    def forward(self, x, e_mask):

        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=e_mask)
        
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


    def forward(self, x, memory, e_mask, d_mask):
        
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x



class TorchModel(ModelBase):
    def __init__(self, config):
        super(TorchModel, self).__init__(config)
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    
    def pad_mask(self, x):
        return x == self.pad_id
    

    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def encode(self, x, e_mask):
        return self.encoder(x, e_mask)


    def decode(self, x, memory, e_mask, d_mask):
        return self.decoder(x, memory, e_mask, d_mask)        