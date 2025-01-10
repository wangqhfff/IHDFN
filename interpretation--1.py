import torch
from torch.utils.data import DataLoader
from dataloader import LoadDataset
from modules import IHDFN
import numpy as np
from Integerization import graph_collate_func
import pandas as pd

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
        'HEADS': 8,
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
        'SEED': 66
    }
}
model = IHDFN(**config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(r'D:\a研究\IHDFN\result\best_model_epoch_83.pth', map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

test_data_path = r'D:\a研究\IHDFN\datasets\interpretation-sample.csv'
test_df = pd.read_csv(test_data_path).dropna()
test_list_IDs = range(len(test_df))

test_data = LoadDataset(test_list_IDs, test_df)
test_loader = DataLoader(test_data, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=False,
                         num_workers=config['SOLVER']['NUM_WORKERS'], collate_fn=graph_collate_func)


def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            v_d, v_p, _ = batch
            v_d = v_d.to(device)
            v_p = v_p.to(device)

            _, _, scores, _ = model(v_d, v_p, mode="eval")
            # 使用 sigmoid 函数将模型输出转换为概率
            probabilities = torch.sigmoid(scores).detach().numpy()

            predictions.extend(probabilities)

    return np.array(predictions)

scores = predict(model, test_loader, device)


print("预测得分：", scores)



