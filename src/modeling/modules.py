import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinear(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.):
        super(NonLinear, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_mask=None):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class AttentionPoolLayers(nn.Module):
    def __init__(self, d_in, d_att, d_out):
        super(AttentionPoolLayers, self).__init__()
        self.q = nn.Parameter(torch.FloatTensor(d_att, 1))
        self.k = nn.Parameter(torch.FloatTensor(d_in, d_att))
        self.v = nn.Parameter(torch.FloatTensor(d_in, d_out))
        self.attn = None
        self.mask = None
        
        self.init_normed()
    
    def init_normed(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.v)
        
    def forward(self, x):
        k = torch.einsum("nxtd,dk->nxkt", x, self.k)
        v = torch.einsum("nxtd,dp->nxpt", x, self.v)
        
        attn_score = torch.einsum("ks,nxkt->nxst", self.q, k)
        
        attn_prob = F.softmax(attn_score, dim=-1)
        self.attn = attn_prob
        
        out = torch.einsum("nxst,nxpt->nxsp", attn_prob, v)
        return out.squeeze(2)
    
class AttentionPool(nn.Module):
    def __init__(self, d_in, d_att, d_out=2):
        super(AttentionPool, self).__init__()
        self.q = nn.Parameter(torch.FloatTensor(d_att, 1))
        self.k = nn.Parameter(torch.FloatTensor(d_in, d_att))
        self.v = nn.Parameter(torch.FloatTensor(d_in, d_out))
        self.attn = None
        self.mask = None
        self.dropout = nn.Dropout(0.15)
        
        self.init_normed()
    
    def init_normed(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.v)
        
    def forward(self, x, x_mask=None):
        k = torch.einsum("ntd,dk->nkt", x, self.k)
        v = torch.einsum("ntd,dp->npt", x, self.v)
        
        attn_score = torch.einsum("ks,nkt->nst", self.q, k)
        
        
        if x_mask is not None:
            self.mask = x_mask.unsqueeze(1)
            attn_score += (x_mask.unsqueeze(1)+1e-15).log()**15
        
        attn_prob = F.softmax(attn_score, dim=-1)
        self.attn = attn_prob
        
        out = torch.einsum("nst,npt->nsp", attn_prob, v)
        return out.squeeze(1)