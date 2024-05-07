import torch
from models.model import FlameGenerator
from torch.utils.data.dataloader import DataLoader
from torch import abs

def test(model:FlameGenerator, loder: DataLoader):
    device = next(model.parameters()).device
    xy_mean = loder.dataset.xy_mean.to(device)
    xy_std = loder.dataset.xy_std.to(device)
    data_mean = loder.dataset.data_mean.to(device)
    data_std = loder.dataset.data_std.to(device)
    with torch.no_grad():
        T_sum = 0
        CO2_sum = 0
        H2O_sum = 0
        size_sum = 0
        count = 0
        for cond, data, size in loder:
            cond = cond.to(device)
            data = data.to(device).permute(0, 2, 3, 1)
            size = size.to(device)
            data_G, size_G = model(cond)
            data_G = data_G.permute(0, 2, 3, 1)
            #
            data = data*data_std + data_mean
            data_G = data_G*data_std + data_mean
            size = size*xy_std + xy_mean
            size_G = size_G*xy_std + xy_mean
            #
            T_sum += abs(data_G[:,:,:,1] - data[:,:,:,1]).sum(dim=0)
            CO2_sum += abs(data_G[:,:,:,4] - data[:,:,:,4]).sum(dim=0)
            H2O_sum += abs(data_G[:,:,:,5]-data[:,:,:,5]).sum(dim=0)
            size_sum += abs(size_G-size).sum(dim=0)
            count += data_G.shape[0]
        T_mean = T_sum.mean() / count
        CO2_mean = CO2_sum.mean() / count
        H2O_mean = H2O_sum.mean() / count
        size_mean = size_sum.mean() / count
    return T_mean, CO2_mean, H2O_mean, size_mean

