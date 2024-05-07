import torch
from models.model import FlameGenerator
from dataloader.dataloader import FlameDataset
from torch.utils.data.dataloader import DataLoader
import yaml
from torch import optim
from test import test

def train(cfg):
    # 加载训练的超参数
    device = cfg['device']
    epochs = cfg['epochs']
    batch = cfg['batch']
    lr = cfg['lr']
    num_worker = cfg['num_worker']
    savepath = cfg['savepath']
    #
    model = FlameGenerator(cfg)
    #
    train_dataset = FlameDataset(cfg, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataset = FlameDataset(cfg, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    #
    opt = optim.Adam(model.parameters(), lr = lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-8)
    #
    for epoch in range(epochs):
        model.to(device).train()
        for cond, data, size in train_loader:
            cond = cond.to(device)
            data = data.to(device)
            size = size.to(device)
            #
            opt.zero_grad()
            data_G,  size_G = model(cond)
            #
            loss = torch.abs(data_G-data).mean() + torch.abs(size_G-size).mean() +\
                5*torch.abs(data_G[:,1,:,:]-data[:,1,:,:]).mean()
            loss.backward()
            opt.step()
        sch.step()
        T_mean, CO2_mean, H2O_mean, size_mean = test(model, test_loader)
        logs = f'\r loss:{loss.item():.5f} T:{T_mean.item():.5f} C:{CO2_mean.item():.5f} H:{H2O_mean.item():.5f} S:{size_mean:.6f} |step:{epoch+1}/{epochs} '
        print(logs, end=' ', flush=True)
        if(T_mean.item()<model.best_metric):
            model.best_metric = T_mean.item()
            model.eval().cpu()
            torch.save(model, savepath)
    print('\n')

if __name__ == '__main__':
    #
    with open("./cfg.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    train(cfg)