import torch
import numpy as np
import os

class Reconstruction:
    def __init__(self, unet, device):
        self.unet = unet
        self.device = device
    
    def __call__(self, x, y0, w):
        def _compute_alpha(t):
            x_0 = x_0.to(self.device)
            betas = np.linspace(0.001, 0.02, 1000, dtype = np.float64)
            betas = torch.tensor(betas).type(torch.float64).to(self.device)
            beta = torch.cat([torch.zeros(1).to(self.device), betas], dim = 0)
            beta = beta.to(self.device)
            a = (1 - beta).cumprod(dim = 0).index_select(0, t+1).view(-1, 1, 1, 1)
            return a
        
        test_trajectoy_steps = torch.Tensor([200]).type(torch.int64).to(self.device).long()
        at = _compute_alpha(test_trajectoy_steps)
        xt = at.sqrt() * x + (1- at).sqrt() * torch.randn_like(x).to(self.device)
        seq = range(0 , 200, 20)
                    
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [xt]
            for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(self.device)
                next_t = (torch.ones(n) * j).to(self.device)
                at = _compute_alpha(t.long())
                at_next = _compute_alpha(next_t.long())
                xt = xs[-1].to(self.device)
                self.unet = self.unet.to(self.device)
                et = self.unet(xt, t)
                yt = at.sqrt() * y0 + (1-at).sqrt() * et
                et_hat = et - (1 - at).sqrt() * w * (yt-xt)
                x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
                c1 = (
                    1 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
                xs.append(xt_next)
        return xs
