from modules import IHDFN
from time import time
from Integerization import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import LoadDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description=" IHDFN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    mkdir(cfg.RESULT.OUTPUT_DIR)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = LoadDataset(df_train.index.values, df_train)
    val_dataset = LoadDataset(df_val.index.values, df_val)
    test_dataset = LoadDataset(df_test.index.values, df_test)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False

    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = IHDFN(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
