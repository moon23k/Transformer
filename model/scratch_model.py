import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import clones, Embeddings, ModelBase



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads

        assert hidden_dim // self.n_heads
        self.head_dim = hidden_dim // self.n_heads
        
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 4)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(config.device)


    def forward(self, query, key, value, mask = None):

        orig_shape = list(query.shape)
        split_shape = [query.size(0), -1, self.n_heads, self.head_dim]

        Q, K, V = [lin(x).view(split_shape).transpose(1, 2) \
                   for lin, x in zip(self.linears, (query, key, value))]   

        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            score = score.masked_fill(mask==0, -1e10)

        attention = torch.softmax(score, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(orig_shape)

        del Q, K, V

        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.fc_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        x = self.dropout(F.gelu(self.fc_1(x)))
        return self.fc_2(x)        



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        hidden_dim = config.hidden_dim
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.pff_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x, e_mask):
        _x = self.self_attn(x, x, x, e_mask)
        x = self.self_attn_norm(x + self.dropout(_x))
        _x = self.pff(x)
        x = self.pff_norm(x + self.dropout(_x))

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

        self.dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x, m, e_mask, d_mask):

        _x = self.self_attn(x, x, x, d_mask)
        x = self.self_attn_norm(x + self.dropout(_x))

        _x = self.enc_attn(x, m, m, e_mask)
        x = self.enc_attn_norm(x + self.dropout(_x))

        _x = self.pff(x)
        x = self.pff_norm(x + self.dropout(_x))

        return x



class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device
        self.emb = Embeddings(config)
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )


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
        self.dropout = nn.Dropout(config.dropout_ratio)

        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )


    def forward(self, x, memory, e_mask, d_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask)
        return x
     


class ScratchModel(ModelBase):
    def __init__(self, config):
        super(ScratchModel, self).__init__(config)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        sz = x.size(1)
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((sz, sz), device=self.device)).bool()
        return pad_mask & sub_mask


    def encode(self, x, e_mask):
        return self.encoder(x, e_mask)


    def decode(self, x, memory, e_mask, d_mask):
        return self.decoder(x, memory, e_mask, d_mask)                
