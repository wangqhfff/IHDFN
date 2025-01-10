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


class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

   # 交叉注意
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


# class MultiheadAttention(Module):
#     r"""Allows the model to jointly attend to information
#     from different representation subspaces.
#     See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
#
#     .. math::
#         \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
#
#     where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
#
#     Args:
#         embed_dim: Total dimension of the model.
#         num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
#             across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
#         dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
#         bias: If specified, adds bias to input / output projection layers. Default: ``True``.
#         add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
#         add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
#             Default: ``False``.
#         kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
#         vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#
#     Examples::
#
#         >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
#     """
#     __constants__ = ['batch_first']
#     bias_k: Optional[torch.Tensor]
#     bias_v: Optional[torch.Tensor]
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#
#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)
#
#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
#
#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None
#
#         self.add_zero_attn = add_zero_attn
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)
#
#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)
#
#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True
#
#         super(MultiheadAttention, self).__setstate__(state)
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
#         r"""
#     Args:
#         query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
#             when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
#             and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
#             key-value pairs to produce the output. See "Attention Is All You Need" for more details.
#         key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
#             ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
#             :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
#         value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
#             ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
#             :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
#         key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
#             to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
#             For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
#             the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
#             value will be ignored.
#         need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
#             Default: ``True``.
#         attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
#             :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
#             :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
#             broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
#             Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
#             corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
#             corresponding position is not allowed to attend. For a float mask, the mask values will be added to
#             the attention weight.
#
#     Outputs:
#         - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
#           :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
#           the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
#         - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
#           size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
#           when ``need_weights=True``.
#         """
#         if self.batch_first:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
#
#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight)
#         else:
#             attn_output, attn_output_weights = F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask)
#         if self.batch_first:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights
