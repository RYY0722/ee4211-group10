# ----------------------------------------
# Written by Xiaoqing GUO
# ----------------------------------------

import torch
import torch.nn as nn
from datasets.BirdDataset import BirdDataset

def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'bird' or dataset_name == 'BIRD':
		return BirdDataset('bird', cfg, period)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
