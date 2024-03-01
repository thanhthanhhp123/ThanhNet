from common import *

import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from backbones import *
from simplenet import *

backbone = GSWideResNet()

patch_maker = PatchMaker(
    3, stride = 1
)

device = "cpu"
layers_to_extract_from = ['layer2', 'layer3']

feature_agg = NetworkFeatureAggregator(
    backbone, layers_to_extract_from, device
)

feature_dim = feature_agg.feature_dimensions(
    (1, 65, 65)
)

print(feature_dim)