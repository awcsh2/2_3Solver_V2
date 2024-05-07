import torch
import torch.nn as nn
import yaml

class DataGenerator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        hiden = cfg['data_hiden']
        input_dim = cfg['ctrl'] + 2
        data_dim = cfg['data_dim']
        layers = []
        last_layer = input_dim
        for layer in hiden:
            layers.append(nn.Conv2d(last_layer, layer, 1, 1, 0))
            layers.append(nn.LeakyReLU())
            last_layer = layer
        layers.append(nn.Conv2d(last_layer, data_dim, 1, 1, 0))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class SizePredictor(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        hiden = cfg['size_hiden']
        ctrl = cfg['ctrl']
        layers = []
        last_layer = ctrl
        for layer in hiden:
            layers.append(nn.Linear(last_layer, layer))
            layers.append(nn.LeakyReLU())
            last_layer = layer
        layers.append(nn.Linear(last_layer, 2))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class FlameGenerator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.G_data = DataGenerator(cfg)
        self.P_size = SizePredictor(cfg)
        #
        mesh_size = cfg['mesh_size']
        ctrl = cfg['ctrl']
        self.mesh = torch.ones([1, 2+ctrl, mesh_size[0], mesh_size[1]])
        self.mesh[:, 0:2, :, :] *= self.create_mesh(mesh_size, [2,2])
        self.best_metric = 1000
    
    def forward(self, ctrl):
        with torch.no_grad():
            ctrl_embed = ctrl.unsqueeze(2).unsqueeze(3)
            mesh = self.mesh.detach().clone().to(ctrl_embed.device)
            mesh = mesh.repeat(ctrl.shape[0], 1, 1, 1)
            mesh[:, 2:, :, :] = mesh[:, 2:, :, :] * ctrl_embed
        size = self.P_size(ctrl)
        data = self.G_data(mesh)
        return data, size

    # 网格生成函数
    def create_mesh(self, mesh_size, scale,  q=1.01):
        h = mesh_size[0]
        w = mesh_size[1]
        mesh_XY = torch.zeros([2, h, w])
        x_scale = scale[0]
        y_scale = scale[1]
        # 初始网格的宽
        ax = x_scale * (1-q) / (1-q**(w-1)) 
        # 初始网格的高
        ay = y_scale * (1-q) / (1-q**(h-1)) 
        # 当前节点在网格坐标系下的坐标
        x = 0
        y = 0
        for i in range(h):
            if i > 0:
                wy = ay * (q**i)
            else:
                wy = 0
            y += wy
            x = 0
            for j in range(w):
                if j > 0:
                    wx = ax * (q**j)
                else:
                    wx = 0
                x += wx
                # 写入数据
                mesh_XY[0, i, j] = x
                mesh_XY[1, i, j] = y
        return mesh_XY.unsqueeze(0)



if __name__ == '__main__':
    with open("/A/Wei/solver/2.3solver-V2/cfg.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    # x = torch.rand([1, 12+2, 100, 500])
    # model = DataGenerator(cfg)
    # y = model(x)
    # print(y.shape)

    ctrl = torch.randn([4, 12])
    model = FlameGenerator(cfg)
    data, size = model(ctrl)
    print(data.shape)
    print(size.shape)


