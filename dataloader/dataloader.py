import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import yaml

def load_dataset(cfg):
    # read txt as list
    txt = cfg['dataset']
    dataset = []
    with open(txt, 'r') as f:
        for row in f.readlines():
            row = row.rstrip()
            row = row.split('\t')
            # row[0]:x row[1]:y row[2]:s
            dataset.append([row[0], row[1], row[2]])
    return dataset

class FlameDataset(Dataset):
    def __init__(self, cfg, train=True) -> None:
        super().__init__()
        dataset = load_dataset(cfg)
        n = cfg['n']
        if train:
            self.data = dataset[0:n]
        else:
            self.data = dataset[0:n]
        self.input_mean = torch.Tensor(cfg['input_mean'])
        self.input_std = torch.Tensor(cfg['input_std'])
        self.xy_mean = torch.Tensor(cfg['xy_mean'])
        self.xy_std = torch.Tensor(cfg['xy_std'])
        self.data_mean = torch.Tensor(cfg['data_mean']).unsqueeze(0).unsqueeze(0)
        self.data_std = torch.Tensor(cfg['data_std']).unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        cond = torch.Tensor(np.load(self.data[index][0]))
        data = torch.Tensor(np.load(self.data[index][1]))
        size = torch.Tensor(np.load(self.data[index][2]))
        cond = (cond - self.input_mean) / self.input_std
        data = (data - self.data_mean) / self.data_std
        data = data.permute(2,0,1)
        size = (size - self.xy_mean) / self.xy_std
        return cond, data, size

if __name__ == '__main__':
    with open("/A/Wei/solver/2.3solver-V2/cfg.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    dataset = FlameDataset(cfg, True)
    loader = DataLoader(dataset, batch_size=1)
    for cond, data, size in loader:
        print(cond.shape)
        print(data.shape)
        print(size.shape)
        break
    