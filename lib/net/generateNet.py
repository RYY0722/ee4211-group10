# ----------------------------------------
# Written by Xiaoqing GUO
# ----------------------------------------

import torch
import torch.nn as nn
from net.deeplabv3plus import deeplabv3plus
from net.denseaspp import DenseASPP

def generate_net(cfg):
	if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus(cfg)
	if cfg.MODEL_NAME == 'denseaspp' or cfg.MODEL_NAME == 'DenseASPP':
		return DenseASPP(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
