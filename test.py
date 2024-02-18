from backbones import ResNet18
from simplenet import Discriminator, Projection
import torch

backbone = ResNet18()
discriminator = Discriminator(in_planes=0)
projection = Projection(in_planes = 2)

output = ResNet18(torch.randn(1, 60, 60))
print(output.shape)