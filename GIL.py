import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from typing import Optional, Tuple
if torch.cuda.is_available():
    device = torch.device('cuda')


class GIL(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, num_heads=4, dropout=0.01, pooling_k=3):
        super(GIL, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.num_heads = num_heads
        self.pooling_k = pooling_k

        self.v_net = FCNet([v_dim, h_dim], dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim], dropout=dropout)

        self.crossAttention = CrossAttention(h_dim, n_heads=8, dropout=0.2)
        self.output_net = FCNet([h_dim, h_out], dropout=dropout)

        # Add Pooling layer
        if 1 < pooling_k:
            self.pooling = nn.AvgPool1d(pooling_k, stride=pooling_k)
        else:
            self.pooling = None

    def forward(self, v, q):
        v = self.v_net(v)
        q = self.q_net(q)
        attn_output, attn_weights = self.crossAttention(q, v, v)
        # Apply Pooling
        if self.pooling is not None:
            attn_output = attn_output.permute(0, 2, 1)
            attn_output = self.pooling(attn_output)
            attn_output = attn_output.permute(0, 2, 1)

        output = self.output_net(attn_output)

        return output, attn_weights




class CrossAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)



        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]
        return x,attention


