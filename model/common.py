import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple



def shift_trg(x):
    return x[:, :-1], x[:, 1:]


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        max_len = config.max_len if config.task != 'sum' else config.max_len * 4
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
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



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




def ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()

        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.pad_id, 
            label_smoothing=0.1
        ).to(self.device)
        
        self.out = namedtuple('Out', 'logit loss')        



    def pad_mask(self):
        pass

    def dec_mask(self):
        pass

    def encode(self):
        return


    def forward(self, src, trg):
        trg, label = shift_trg(trg)
        
        e_mask = self.pad_mask(src) 
        d_mask = self.dec_mask(trg)

        memory = self.encode(src, e_mask)
        dec_out = self.decode(trg, memory, e_mask, d_mask)
        logit = self.generator(dec_out)

        self.out.logit = logit
        
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len))
        pred = pred.to(torch.long).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.pad_mask(x)
        memory = self.encode(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_mask = self.dec_mask(y)
            d_out = self.decode(y, memory, e_mask, d_mask)

            logit = self.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[-1:]

        return pred