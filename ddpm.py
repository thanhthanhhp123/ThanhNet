import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


timesteps = 16
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)

class Block(nn.Module):
    def __init__(self, in_channels = 128, size = 28):
        super(Block, self).__init__()

        self.conv_param = nn.Conv2d(in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.dense_ts = nn.Linear(128, 128)

        self.layer_norm = nn.LayerNorm([128, size, size])

    
    def forward(self, x, x_ts):
        x_param = F.relu(self.conv_param(x))

        time_param = F.relu(self.dense_ts(x_ts))
        time_param = time_param.view(time_param.size(0), 128, 1, 1)
        x_param = x_param * time_param

        x_out = F.relu(self.conv_out(x))
        x_out = x_out + x_param
        x_out = F.relu(self.layer_norm(x_out))

        return x_out
    

class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()

        self.l_ts = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.down_x28 = Block(in_channels=1536, size=28)
        self.down_x14 = Block(size=14)
        self.down_x7 = Block(size=7)

        self.mlp = nn.Sequential(
            nn.Linear(6400, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 28 * 7 * 7),
            nn.LayerNorm(28 * 7 * 7),
            nn.ReLU(),
        )

        self.up_x7 = Block(in_channels=28 + 128, size = 7)
        self.up_x14 = Block(in_channels=256, size = 14)
        self.up_x28 = Block(in_channels=256, size = 28)

        self.cnn_out = nn.Conv2d(in_channels=128, out_channels=1536, kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_ts):
        x_ts = self.l_ts(x_ts)

        blocks = [
            self.down_x28,
            self.down_x14,
            self.down_x7,
        ]

        x_left_layers = []
        for i, block in enumerate(blocks):
            x = block(x, x_ts)
            x_left_layers.append(x)
            if i < len(blocks) - 1:
                x = F.avg_pool2d(x, 2)

        x = x.view(-1, 128 * 7 * 7)
        x = torch.cat([x, x_ts], dim=1)
        x = self.mlp(x)
        x = x.view(-1, 28, 7, 7)

        blocks = [
            self.up_x7,
            self.up_x14,
            self.up_x28,
        ]

        for i, block in enumerate(blocks):
            x_left = x_left_layers[len(blocks) - i - 1]
            x = torch.cat([x, x_left], dim=1)

            x = block(x, x_ts)
            if i < len(blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = self.cnn_out(x)
        return x
    
    def forward_noise(self, x, t):
        a = time_bar[t]
        b = time_bar[t + 1]

        noise = np.random.normal(0, 1, x.shape)
        a = a.reshape((-1, 1, 1, 1))
        b = b.reshape((-1, 1, 1, 1))

        x = x.to('cpu').detach().numpy()

        img_a = x * (1 - a) + noise * a
        img_b = x * (1 - b) + noise * b

        return img_a, img_b
    
    def generate_ts(self, num):
        return np.random.randint(0, timesteps, num)

# model = DDPM()
# model.eval()
# x_ts = model.generate_ts(32)
# x = torch.randn(32, 1536, 28, 28)
# x_a, x_b = model.forward_noise(x, x_ts)
# x_ts = torch.from_numpy(x_ts).view(-1, 1).float()
# x_a = x_a.float()
# x_b = x_b.float()

# y_p = model(x_a, x_ts)

# print(y_p.shape)




