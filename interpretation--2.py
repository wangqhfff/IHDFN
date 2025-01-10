import torch
from torch.utils.data import DataLoader
from dataloader import LoadDataset
from modules import IHDFN
import numpy as np
from Integerization import graph_collate_func
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        'SEED': 666
    }
}
model = IHDFN(**config)

state_dict = torch.load(r'D:\pycharm\IHDFN\result\best_model_epoch_15.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

test_data_path = r'D:\pycharm\IHDFN\datasets\interpretation-sample.csv'
test_df = pd.read_csv(test_data_path).dropna()
test_list_IDs = range(len(test_df))

test_data = LoadDataset(test_list_IDs, test_df)
test_loader = DataLoader(test_data, batch_size=config['SOLVER']['BATCH_SIZE'], shuffle=False,
                         num_workers=config['SOLVER']['NUM_WORKERS'], collate_fn=graph_collate_func)

def get_att_values_and_features(model, data_loader, device):
    model.eval()
    att_values = []

    with torch.no_grad():
        for batch in data_loader:
            v_d, v_p, _ = batch
            v_d = v_d.to(device)
            v_p = v_p.to(device)
            # 确保模型也在同一个设备上
            model = model.to(device)
            _, _, _, att = model(v_d, v_p, mode="eval")

            att = att.detach().cpu().numpy()
            att_values.extend(att)

    return att_values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
att_values = get_att_values_and_features(model, test_loader, device)
att_values = np.array(att_values)


np.save("att_values.npy", att_values)


def get_top_percentile_indices(att_values, percentile=20):
    """Get the indices of the top percentile attention values."""
    # Compute the number of top elements to select
    num_top_elements = int(np.ceil(percentile / 100 * att_values.size))

    # Get the indices that would sort the array
    sorted_indices = np.argsort(att_values)

    # Select the indices of the top elements
    top_indices = sorted_indices[-num_top_elements:]

    return top_indices

# Load the saved att_values
att_values = np.load("att_values.npy")

for index in range(len(test_df)):
    # Select the appropriate row of the att_values matrix
    current_att_values = att_values[index, :, :]

    # Load the SMILES code from the interpretation-sample.csv file
    smiles_code = test_df.iloc[index, 0]

    # Create a molecule object from the SMILES code
    mol = Chem.MolFromSmiles(smiles_code)

    # Get the number of actual atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # Get the attention weights for the actual atoms only
    actual_att_weights = current_att_values.flatten()[:num_atoms]

    # Find the indices for the top 20% important attention values
    important_indices = get_top_percentile_indices(actual_att_weights, percentile=20)

    # Define the highlight colors for atoms
    highlight_colors = {}
    for i in important_indices:
        # Convert numpy.int64 to int
        i = int(i)
        highlight_colors[i] = (0.5, 1, 0.5, 1)

    # Display the molecule with highlighted atoms
    img = Draw.MolToImage(mol, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors, size=(1024, 1024))
    img.show()
    img.save(f"highlighted_molecule_{index}.png")