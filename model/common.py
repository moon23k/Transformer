import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        max_len = config.max_len if config.task != 'summarization' else config.max_len * 4
        pe = torch.zeros(max_len, config.emb_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.pos_dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)
            self.fc_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_dropout(self.pos_emb(out))

        if not self.use_fc_layer:
            return out
        return self.fc_dropout(self.fc(out))



class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))





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