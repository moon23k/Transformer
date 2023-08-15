import math, torch
import torch.nn as nn
from model.common import (
    clones, 
    Embeddings,
    LayerNorm, 
    PositionwiseFeedForward, 
    SublayerConnection,
    ModelBase
)



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e-4)
    
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value)



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config.hidden_dim % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.linears = clones(nn.Linear(config.hidden_dim, config.hidden_dim), 4)


    def forward(self, query, key, value, mask=None):
        #orig_shape: [ batch_size, seq_len, hidden_dim ]
        #attn_shape: [ batch_size, seq_len, n_heads, head_dim ]
        
        orig_shape = list(query.shape)
        split_shape = [query.size(0), -1, self.n_heads, self.head_dim]

        query, key, value = [lin(x).view(split_shape).transpose(1, 2) \
                            for lin, x in zip(self.linears, (query, key, value))]       
        
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (x.transpose(1, 2).contiguous().view(orig_shape))

        del query, key, value
        return self.linears[-1](x)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.src_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, memory, e_mask, d_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, d_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, e_mask))
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



class BaseModel(ModelBase):
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)


    def encode(self, x, e_mask):
        return self.encoder(x, e_mask)

    def decode(self, x, memory, e_mask, d_mask):
        return self.decoder(x, memory, e_mask, d_mask)