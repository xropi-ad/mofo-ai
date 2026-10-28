#! /bin/python3

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transform as T


dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"dev: {dev}")
data_dir = "/mount/+data/cifar10"






















