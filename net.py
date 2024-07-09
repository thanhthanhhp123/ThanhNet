import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

import common

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x
    
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.input_shape = input_shape
        self.patch_maker = PatchMaker(3, stride=1)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_agg = common.NetworkFeatureAggregator(
            backbone, layers_to_extract_from, device
        )
        feature_dimensions = feature_agg.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_agg

        preprocessing = common.PreProcessing(feature_dimensions, 1536)
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = common.Aggregator(target_dim=1536)

        _ = preadapt_aggregator.to(device)
        self.forward_modules["preadapt_aggregtor"] = preadapt_aggregator

        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, x):
        features = self.forward_modules["feature_aggregator"](x)
        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1],
                *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1,
                                          *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregtor"](features)

        features = features.reshape(-1, 28, 28, 1536)
        features = torch.permute(features, (0, 3, 1, 2))

        return features, patch_shapes

class Block(nn.Module):
    def __init__(self, in_channels=128, size=32):
        super(Block, self).__init__()
        
        self.conv_param = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)

        self.dense_ts = nn.Linear(192, 128)

        self.layer_norm = nn.LayerNorm([128, size, size])

    def forward(self, x_img, x_ts):
        x_parameter = F.relu(self.conv_param(x_img))

        time_parameter = F.relu(self.dense_ts(x_ts))
        time_parameter = time_parameter.view(-1, 128, 1, 1)
        x_parameter = x_parameter * time_parameter

        x_out = self.conv_out(x_img)
        x_out = x_out + x_parameter
        x_out = F.relu(self.layer_norm(x_out))

        return x_out

class DDPMs(nn.Module):
    def __init__(self, input_size=(32, 32, 3), time_steps = 16, device = 'cuda' if torch.cuda.is_available() else 'cpu' ):
        super(DDPMs, self).__init__()

        self.input_size = input_size
        self.time_steps = time_steps
        self.device = device
        self.time_bar = 1 - np.linspace(0, 1, time_steps + 1)

        self.l_ts = nn.Sequential(
            nn.Linear(1, 192),
            nn.LayerNorm([192]),
            nn.ReLU(),
        )
        
        self.down_x32 = Block(in_channels=3, size=32)
        self.down_x16 = Block(size=16)
        self.down_x8 = Block(size=8)
        self.down_x4 = Block(size=4)
        
        self.mlp = nn.Sequential(
            nn.Linear(2240, 128),
            nn.LayerNorm([128]),
            nn.ReLU(),
            
            nn.Linear(128, 32 * 4 * 4), # make [-1, 32, 4, 4]
            nn.LayerNorm([32 * 4 * 4]),
            nn.ReLU(),
        )
        
        self.up_x4 = Block(in_channels=32 + 128, size=4)
        self.up_x8 = Block(in_channels=256, size=8)
        self.up_x16 = Block(in_channels=256, size=16)
        self.up_x32 = Block(in_channels=256, size=32)
        
        self.cnn_output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, padding=0)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x, x_ts):
        x_ts = self.l_ts(x_ts)

        blocks = [
            self.down_x32,
            self.down_x16,
            self.down_x8,
            self.down_x4,]
        x_left_layers = []
        for i, block in enumerate(blocks):
            x = block(x, x_ts)
            x_left_layers.append(x)
            if i < len(blocks) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(-1, 128 * 4 * 4)
        x = torch.cat([x, x_ts], dim=1)
        x = self.mlp(x)
        x = x.view(-1, 32, 4, 4)

        blocks = [
            self.up_x4,
            self.up_x8,
            self.up_x16,
            self.up_x32,]
        
        for i, block in enumerate(blocks):
            x_left = x_left_layers[len(blocks) - i - 1]
            x = torch.cat([x, x_left], dim=1)

            x = block(x, x_ts)
            if i < len(blocks) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = self.cnn_output(x)
        return x
    
    def forward_noise(self, x, t):
        a = self.time_bar[t]
        b = self.time_bar[t + 1]

        noise = np.random.normal(size=x.shape)
        a = a.reshape((-1, 1, 1, 1))
        b = b.reshape((-1, 1, 1, 1))

        img_a = x * (1-a) + noise * a
        img_b = x * (1-b) + noise * b
        return img_a, img_b
    
    def generate_ts(self, num):
        return np.random.randint(0, self.time_steps, num)