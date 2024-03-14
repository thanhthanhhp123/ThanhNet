import torch
import torch.nn as nn
import torch.nn.functional as F
# from dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, s=1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(      
            nn.Conv2d(1, 4*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),

            nn.Conv2d(4*s, 8*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*s),
            nn.ReLU(),

            nn.Conv2d(8*s, 16*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16*s),
            nn.ReLU(),

            nn.Conv2d(16*s, 32*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*s),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, s=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(32*s, 16*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16*s),
            nn.ReLU(),

            nn.Conv2d(16*s, 8*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*s),
            nn.ReLU(),

            nn.Conv2d(8*s, 4*s, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),

            nn.Conv2d(4*s, 1, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    ae =  AutoEncoder()
    random_input = torch.rand(1, 1, 65, 65)
    output = ae(random_input)



