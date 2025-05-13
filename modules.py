import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from GIL import GIL
from torch.nn.utils.weight_norm import weight_norm
import math

import numpy as np
import dgl.nn as dglnn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#conv_filters1 = [[1,32],[3,32],[5,64]]
conv_filters1 = [[1,32],[3,32],[5,64],[7,128]]
embedding_size1 = output_dim1 = 128
d_ff = 256
n_heads = 8
d_k = 16
n_layer = 1

smi_vocab_size = 65
seq_vocab_size = 26

seed = 990721
embed_dim=128

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()
def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
class IHDFN(nn.Module):
    def __init__(self, **config):
        super(IHDFN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        self_attention_heads = config["SELF_ATTENTION"]["HEADS"]
        self_attention_h_dim = config["SELF_ATTENTION"]["HIDDEN_DIM"]
        self_attention_h_out = config["SELF_ATTENTION"]["OUT_DIM"]

        self.drug_extractor = DrugGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor1 = Seq_Encoder()
        self.protein_extractor2 = nanoGPT(vocab_size=26, n_embd=128, n_head=4, n_layer=2, block_size=1200, # n_head=4
                                attention_class=MultiHeadDiffAttention)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.layer_norm3 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # GELU 激活函数\

        self.self_attention = GIL(v_dim=drug_hidden_feats[-1], q_dim=protein_emb_dim, h_dim=self_attention_h_dim, h_out=self_attention_h_out, num_heads=self_attention_heads)
        mlp_in_dim = self_attention_h_out

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p1,v_p298 = self.protein_extractor1(v_p)
        v_p2 = self.protein_extractor2(v_p298)
        # starnet
        v_p11 = self.layer_norm1(v_p1)
        v_p12 = self.gelu(v_p11)
        v_p22 = self.layer_norm2(v_p2)
        v_p22 = self.gelu(v_p22)
        v_p = v_p12 * v_p22
        v_p = self.layer_norm3(v_p + v_p11)
        f, att = self.self_attention(v_d, v_p)
        f_pooled = torch.mean(f, dim=1)  # Global max pooling
        score = self.mlp_classifier(f_pooled)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


# MLP
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Transformer Block
class Block3(nn.Module):
    def __init__(self, n_embd, n_head, attention_class, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = attention_class(n_embd, n_head, layer_idx)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 动态编码
class ConvEmbedding1(nn.Module):
    def __init__(self, vocab_size, embedding_size1, conv_filters, output_dim1, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size1)

        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size1, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size1, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        # The dimension of concatenated vectors obtained from multiple one-dimensional convolutions
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim1)


    def forward(self, inputs):
        inputs = inputs.long()
        embeds = self.embed(inputs).transpose(-1,-2) # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2) # (batch_size, seq_len, num_filters)
        res_embed = self.projection(res_embed)
        return res_embed

# A highway neural network, where the dimensions of input and output are the same, similar to the ResNet principle
class Highway1(nn.Module):
    def __init__(self, input_dim, num_layers, acticvation = F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([nn.Linear(input_dim, input_dim*2) for _ in range(num_layers)])
        self.acticvation = acticvation
        for layer in self.layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            # The output dimension is 2 * input_ Dim, the first half is used for hidden layer output, and the second half is used for gate output
            hidden = self.acticvation(projected_inputs[:,:self.input_dim])
            gate = torch.sigmoid(projected_inputs[:,self.input_dim:])
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs

class SelfAttention1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim = - 1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention1(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(embedding_size1, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(embedding_size1, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embedding_size1, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, embedding_size1, bias=False)
        self.ln = nn.LayerNorm(embedding_size1)
        self.attn = SelfAttention1()
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = self.attn(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_k) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.ln(output)


class FeedForward1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size1, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embedding_size1, bias=False)
        )
        self.ln = nn.LayerNorm(embedding_size1)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output) # [batch_size, seq_len, d_model]


class EncoderLayer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi = MultiHeadAttention1()
        self.feed = FeedForward1()

    def forward(self, en_input, attn_mask):
        context = self.multi(en_input, en_input, en_input, attn_mask)
        output = self.feed(context+en_input)
        return output


class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = ConvEmbedding1(seq_vocab_size, embedding_size1, conv_filters1, output_dim1, 'seq')
        self.poc_emb = ConvEmbedding1(seq_vocab_size, embedding_size1, conv_filters1, output_dim1, 'poc')
        self.highway = Highway1(embedding_size1, n_layer)
        self.layers  = nn.ModuleList([EncoderLayer1() for _ in range(n_layer)])
    def forward(self, seq_input):
        output_emb = self.seq_emb(seq_input)
        enc_self_attn_mask = get_attn_pad_mask(seq_input, seq_input)
        for layer in self.layers:
            output_emb1 = layer(output_emb,enc_self_attn_mask)
        return output_emb1,output_emb

# 加残差连接和层归一化
class DrugGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(DrugGCN, self).__init__()
        # 初始特征转换
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        # 定义 GCN 层
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)

        # 定义输出特征维度
        self.output_feats = hidden_feats[-1]

        # 为每一层残差连接后添加层归一化
        self.layer_norm = nn.LayerNorm(self.output_feats)

    def forward(self, batch_graph):
        # 获取原始节点特征
        node_feats = batch_graph.ndata.pop('h')

        # 初始特征转换
        node_feats_transformed = self.init_transform(node_feats)

        # GCN 传播
        node_feats_gnn = self.gnn(batch_graph, node_feats_transformed)

        # 残差连接
        node_feats_residual = node_feats_transformed + node_feats_gnn

        # 添加层归一化
        node_feats_normalized = self.layer_norm(node_feats_residual)

        # 获取批量大小并调整特征维度
        batch_size = batch_graph.batch_size
        node_feats_normalized = node_feats_normalized.view(batch_size, -1, self.output_feats)

        return node_feats_normalized

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits



def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent
