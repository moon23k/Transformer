import torch
import torch.nn as nn
from collections import namedtuple
from model.common import (
    clones, 
    shift_trg, 
    Embeddings, 
    LayerNorm,
    PositionwiseFeedForward, 
    SublayerConnection
)




class MHA(nn.Module):
    def __init__(self, config):
        super(MHA, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, 
            num_heads=config.n_heads, 
            dropout=config.dropout_ratio,
            batch_first=True
            )        

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        out, _ = self.attn(
            query=q, key=k, value=v,
            key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask
            )
        
        return out





class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MHA(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(q=x, k=x, v=x, 
                                                         key_padding_mask=mask))
        return self.sublayer[1](x, self.feed_forward)



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MHA(config)
        self.src_attn = MHA(config)        
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, memory, e_mask, d_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(q=x, k=x, v=x, 
                                                         attn_mask=d_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(q=x, k=m, v=m, 
                                                        key_padding_mask=e_mask))
        return self.sublayer[2](x, self.feed_forward)



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



class H_Transformer(nn.Module):
    def __init__(self, config):
        super(H_Transformer, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=0.1).to(self.device)
        self.out = namedtuple('Out', 'logit loss')


    def pad_mask(self, x):
        return x == self.pad_id


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, src, trg):
        trg, label = shift_trg(trg)

        #Masking
        src_pad_mask = self.pad_mask(src)
        trg_mask = self.dec_mask(trg)

        memory = self.encoder(src, src_pad_mask)
        dec_out = self.decoder(trg, memory, src_pad_mask, trg_mask)
        logit = self.generator(dec_out)

        self.out.logit = logit
        self.out.loss = self.criterion(logit.contiguous().view(-1, self.vocab_size), 
                                       label.contiguous().view(-1))
        return self.out