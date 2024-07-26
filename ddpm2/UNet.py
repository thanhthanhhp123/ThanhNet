import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        '''
        Apply the module to `x` given `emb` timestep embeddings
        '''


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, embed):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, embed)
            else:
                x = layer(x)

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale = 1):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = x.dtype) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb
    

class DownSample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels = None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels

        if use_conv:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, time_embed = None):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels = None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        out_channels = out_channels or in_channels

        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x, time_embed = None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if self.use_conv:
            x = self.conv(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_heads = 1, n_head_channels = -1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, in_channels)

        if n_head_channels == -1:
            self.num_heads = n_heads
        
        else:
            self.num_heads = in_channels // n_head_channels
        
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, kernel_size = 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, kernel_size = 1))
    
    def forward(self, x, time = None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    
    def forward(self, qkv, time = None):
        bs, w, l = qkv.shape
        ch = w // (3 * self.n_heads)
        q, k , v = qkv.reshape(bs * self.n_heads, ch * 3, l).split(ch, dim = 1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim = -1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, l)
    
class ResBlock(TimestepBlock):
    def __init__(self,
                 in_channels,
                 time_embed_dim, 
                 dropout,
                 out_channels = None,
                 use_conv = False,
                 up = False,
                 down = False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
                GroupNorm32(32, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                )
        self.updown = up or down

        if up:
            self.h_upd = UpSample(in_channels, False)
            self.x_upd = UpSample(in_channels, False)
        
        elif down:
            self.h_upd = DownSample(in_channels, False)
            self.x_upd = DownSample(in_channels, False)
        
        else:
            self.h_upd = self.x_upd = nn.Identity()

        
        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            )
        
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNetModel(nn.Module):
    def __init__(self, input_size, base_channels, conv_resample = True,
                 n_heads = 1, n_head_channels = -1, channel_mults = '',
                 num_res_blocks = 2, dropout = 0, attention_resolutions = '28, 14, 7',
                 biggan_updown = True, in_channels = 1536):
        self.dtype = torch.float32
        super().__init__()
        if channel_mults == '':
            channel_mults = (1, 2, 3)
        
        attention_ds = []
        for res in attention_resolutions.split(','):
            attention_ds.append(input_size // int(res))

        self.input_size = input_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels, 1),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))]
                )
        channels = [ch]
        ds = 1
        for i, mult in enumerate(channel_mults):
            # out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout,
                        )]
                ch = base_channels * mult
                # channels.append(ch)

                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(
                                        ch,
                                        time_embed_dim=time_embed_dim,
                                        out_channels=out_channels,
                                        dropout=dropout,
                                        down=True
                                        )
                                if biggan_updown
                                else
                                DownSample(ch, conv_resample, out_channels=out_channels)
                                )
                        )
                ds *= 2
                ch = out_channels
                channels.append(ch)
        self.middle = TimestepEmbedSequential(
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        ),
                AttentionBlock(
                        ch,
                        n_heads=n_heads,
                        n_head_channels=n_head_channels
                        ),
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        )
                )
        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                            ch + inp_chs,
                            time_embed_dim=time_embed_dim,
                            out_channels=base_channels * mult,
                            dropout=dropout
                            )
                    ]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels
                                    ),
                            )

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                            ResBlock(
                                    ch,
                                    time_embed_dim=time_embed_dim,
                                    out_channels=out_channels,
                                    dropout=dropout,
                                    up=True
                                    )
                            if biggan_updown
                            else
                            UpSample(ch, conv_resample, out_channels=out_channels)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
                zero_module(nn.Conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1))
                )
        

    def forward(self, x, time):

        time_embed = self.time_embedding(time)

        skips = []

        h = x.type(self.dtype)
        for i, module in enumerate(self.down):
            h = module(h, time_embed)
            skips.append(h)
        h = self.middle(h, time_embed)
        for i, module in enumerate(self.up):
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)


if __name__ == '__main__':
    model = UNetModel(
        28, 64, dropout=0.2,
        n_heads = 4, attention_resolutions='28,14,7',
        in_channels=1536
    )

    x = torch.randn(1, 1536, 28, 28)
    t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
    print(model(x, t_batch).shape)