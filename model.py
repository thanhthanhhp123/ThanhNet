import torch.nn as nn
import torch
import torch.nn.functional as F
import os

import utils
import net
import students
import common
from ddpm import DDPM

class ThanhNet(nn.Module):
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 train_backbone = False, epochs = 160):
        super(ThanhNet, self).__init__()
        self.pdn = students.get_pdn_small(padding = True)
        self.ddpm = DDPM()
        self.pdn.load_state_dict(torch.load('d_models/pdn.pth'))
        self.pdn.to(device)
        self.ddpm.to(device)

        self.train_backbone = train_backbone
        self.device = device

        self.ddpm_opt = torch.optim.Adam(self.ddpm.parameters(), lr = 1e-4, weight_decay=1e-5)

        if train_backbone:
            self.pdn_opt = torch.optim.Adam(self.pdn.parameters(), lr = 1e-4, weight_decay=1e-5)
    
    def set_model_dir(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _train_ddpm(self, x):
        if self.train_backbone:
            self.pdn.train()
        else:
            self.pdn.eval()
        self.ddpm.train()
        x_ts = self.ddpm.generate_ts(len(x))
        x_a, x_b = self.ddpm.forward_noise(x, x_ts)

        x_ts = torch.from_numpy(x_ts).view(-1, 1).float().to(self.device)
        x_a = x_a.float().to(self.device)
        x_b = x_b.float().to(self.device)
        
        y_p = self.ddpm(x_a, x_ts)
        loss = torch.mean(torch.abs(y_p - x_b))
        self.ddpm_opt.zero_grad()
        loss.backward()
        self.ddpm_opt.step()

    def _inference(self, images):
        self.pdn.eval()
        self.ddpm.eval()
        with torch.no_grad():
            images = images.to(self.device)
            features_map = self.pdn(images)
            reconstructed_feature_map = self.ddpm(features_map)
            

    
    def train(self, training_loader):
        pass

    



