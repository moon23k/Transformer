import torch, copy, math
import torch.nn as nn



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e-4)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        
        self.register_buffer("sequence_pe", self.generate_pe(config.emb_dim))
        self.register_buffer("context_pe", self.generate_pe(config.hidden_dim))

        self.dropout = nn.Dropout(config.dropout_ratio)


    @staticmethod
    def generate_pe(pos_dim, max_len=500):
        pe = torch.zeros(max_len, pos_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2) * -(math.log(10000.0) / pos_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


    def forward(self, x):
        #In case of Encoder Sequence Positional Encoding
        if (x.dim() == 4) and (x.size(-1) == self.emb_dim):
            batch_size, seq_num, seq_len, emb_dim = x.shape
            pos = self.sequence_pe[:, :seq_len].requires_grad_(False)
            x = x.view(batch_size * seq_num, seq_len, emb_dim) + pos
            x = x.view(batch_size, seq_num, seq_len, emb_dim)
        
        #In case of Encoder Context Positional Encoding | x: [batch_size, seq_num, hidden_dim]
        elif (x.dim() != 4) and (x.size(-1) == self.hidden_dim):
            pos = self.context_pe[:, :x.size(1)].requires_grad_(False)    
            x = x + pos

        #In case of Decoder Sequence Positional Encoding | x: [batch_size, seq_len, emb_dim]
        elif (x.dim() != 4) and (x.size(-1) == self.emb_dim):
            pos = self.sequence_pe[:, :x.size(1)].requires_grad_(False)    
            x = x + pos        
        
        return self.dropout(x) 



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)
        self.pos_encoding = PositionalEncoding(config)
        self.fc = nn.Linear(config.emb_dim, config.hidden_dim)

    def forward(self, x):
        out = self.lut(x) * self.scale
        out = self.pos_encoding(out)
        return self.fc(out)



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config.hidden_dim % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        
        self.attn = None
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.linears = clones(nn.Linear(config.hidden_dim, config.hidden_dim), 4)

    def forward(self, query, key, value, mask=None):
        orig_shape = list(query.shape)

        if query.dim() == 4:
            #[batch_size, seq_num, seq_len, n_heads, head_dim]
            split_shape = [query.size(0), query.size(1), -1, self.n_heads, self.head_dim]
        else:
            #[batch_size, seq_len, n_heads, head_dim]
            split_shape = [query.size(0), -1, self.n_heads, self.head_dim]

        query, key, value = [lin(x).view(split_shape).transpose(-2, -3)
                            for lin, x in zip(self.linears, (query, key, value))]       
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (x.transpose(-2, -3).contiguous().view(orig_shape))
        del query, key, value
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



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

