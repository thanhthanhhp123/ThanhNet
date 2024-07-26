import torch
import torch.nn as nn
import numpy as np

def get_loss(model, x_0, t):
    x_0 = x_0.to(model.device)
    betas = np.linspace(0.001, 0.02, 1000, dtype = np.float64)
    b = torch.tensor(betas).type(torch.float64).to(model.device)
    e = torch.randn_like(x_0).to(model.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

    x = at.sqrt() * x_0 + (1-at).sqrt() * e
    output = model(x, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim = 0)

