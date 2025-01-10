import torch
from torch.utils.data import DataLoader
from dataloader import LoadDataset
from modules import MolLoG
import pandas as pd
import numpy as np
from Integerization import graph_collate_func
import heapq


config = {
    'DRUG': {
        'NODE_IN_FEATS': 75,
        'PADDING': True,
        'HIDDEN_LAYERS': [128, 128, 128],
        'NODE_IN_EMBEDDING': 128,
        'MAX_NODES': 290
    },
    'PROTEIN': {
        'NUM_FILTERS': [128, 128, 128],
        'KERNEL_SIZE': [3, 6, 9],
        'EMBEDDING_DIM': 128,
        'PADDING': True
    },
    'SELF_ATTENTION': {
        'HEADS': 4,
        'HIDDEN_DIM': 256,
        'H_OUT': 256,
        'OUT_DIM': 128,
        'PADDING': True
    },
    'DECODER': {
        'NAME': 'MLP',
        'IN_DIM': 256,
        'HIDDEN_DIM': 512,
        'OUT_DIM': 128,
        'BINARY': 1
    },
    'SOLVER': {
        'MAX_EPOCH': 100,
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 0,
        'LR': 5e-05,
        'DA_LR': 0.001,
        'SEED': 42
    }
}
model = MolLoG(**config)

state_dict = torch.load(r'D:\a研究\KDMHA\result\best_model_epoch_83.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

test_data_path = r'D:\a研究\KDMHA\datasets\interpretation-sample.csv'
test_df = pd.read_csv(test_data_path).dropna()
test_list_IDs = range(len(test_df))

test_data = LoadDataset(test_list_IDs, test_df)
test_loader = DataLoader(test_data, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=False,
                         num_workers=config['SOLVER']['NUM_WORKERS'], collate_fn=graph_collate_func)

def get_att_values_and_features(model, data_loader, device):
    model.eval()
    att_values = []
    vd_values = []
    vp_values = []

    with torch.no_grad():
        for batch in data_loader:
            v_d, v_p, _ = batch
            v_d = v_d.to(device)
            v_p = v_p.to(device)

            v_d_out, v_p_out, _, att = model(v_d, v_p, mode="eval")

            att = att.detach().cpu().numpy()
            v_d_out = v_d_out.detach().cpu().numpy()
            v_p_out = v_p_out.detach().cpu().numpy()

            att_values.extend(att)
            vd_values.extend(v_d_out)
            vp_values.extend(v_p_out)

    return att_values, vd_values, vp_values


state_dict = torch.load(r'D:\a研究\KDMHA\result\best_model_epoch_83.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

test_data_path = r'D:\a研究\KDMHA\datasets\interpretation-sample.csv'
test_df = pd.read_csv(test_data_path).dropna()
test_list_IDs = range(len(test_df))

test_data = LoadDataset(test_list_IDs, test_df)
test_loader = DataLoader(test_data, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=False,
                         num_workers=config['SOLVER']['NUM_WORKERS'], collate_fn=graph_collate_func)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(test_data_path)
test_list_IDs = test_df.index.tolist()

test_data = LoadDataset(test_list_IDs, test_df, max_drug_nodes=290)
test_loader = DataLoader(test_data, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=False,
                          num_workers=config['SOLVER']['NUM_WORKERS'], collate_fn=graph_collate_func)

att_values, vd_values, vp_values = get_att_values_and_features(model, test_loader, device)
np.save("att_values.npy", att_values)
np.save("vd_values.npy", vd_values)
np.save("vp_values.npy", vp_values)

print("att_values shape:", np.array(att_values).shape)
print("vd_values shape:", np.array(vd_values).shape)
print("vp_values shape:", np.array(vp_values).shape)

# Convert att_values to a NumPy array
att_values = np.array(att_values)

# 获取att中最大的10个权重值及其对应的位置
att_flattened = att_values.reshape(-1)
top_10_indices = heapq.nlargest(10, range(len(att_flattened)), att_flattened.take)
top_10_values = att_flattened[top_10_indices]

# 将位置转换为att矩阵中的二维坐标
top_10_coords = [divmod(index, 290) for index in top_10_indices]

# 获取对应的vd和vp值
top_10_vd = [vd_values[0][coord[1]] for coord in top_10_coords]
top_10_vp = [vp_values[0][coord[0]] for coord in top_10_coords]

# 打印结果
for i in range(10):
     print(f"Top {i+1}:")
     print(f"Value: {top_10_values[i]}")
     print(f"Position in att matrix: {top_10_coords[i]}")
     print(f"vd: {top_10_vd[i]}")
     print(f"vp: {top_10_vp[i]}\n")

# Find the corresponding drug and protein sequences
# drug_index = 1091
# protein_index = 8
#
# original_drug_smiles = None
# original_protein_sequence = None
# for i, (v_d, v_p, y) in enumerate(test_loader):
#     if i == drug_index:
#         original_drug_smiles = v_d
#     if i == protein_index:
#         original_protein_sequence = v_p
#     if original_drug_smiles is not None and original_protein_sequence is not None:
#         break
#
# print("Original Drug SMILES:", original_drug_smiles)
# print("Original Protein Sequence:", original_protein_sequence)
#
#
#
#
#
#
#
# # Get the original vd and vp nodes
# original_vd_node = original_graph.filter_nodes(lambda nodes: nodes.data['type'] == 0)
# original_vp_node = original_graph.filter_nodes(lambda nodes: nodes.data['type'] == 1)
#
# # Get the node features for the specific vd and vp nodes
# original_vd_feature = original_vd_node.ndata['h'][8].cpu().numpy()
# original_vp_feature = original_vp_node.ndata['h'][1091 - original_graph.batch_num_nodes()[0]].cpu().numpy()
#
# print("Original vd feature:", original_vd_feature)
# print("Original vp feature:", original_vp_feature)


#
#
# # 1. 从 test_df 提取真实标签值
# true_labels = test_df.iloc[:, 2].values
#
# # 2. 使用 predict 函数生成预测结果
# predictions = predict(model, test_loader, device)
#
# # 3. 比较预测结果与真实标签，并计算准确度
# accuracy = np.sum(predictions.flatten() == true_labels) / len(true_labels)
#
# print("预测准确度：", accuracy)



