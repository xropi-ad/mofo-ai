#! /opt/venv/bin/python3

import torch as t
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

t.manual_seed(0)    # to make it deterministic
dev = "cuda" if t.cuda.is_available() else "cpu"

with t.no_grad():
    lmodel = nn.Sequential( nn.Linear(1, 256),
                           nn.ReLU(),
                           nn.Linear(256, 256),
                           nn.ReLU(),
                           nn.Linear(256, 1)
    )
    lmodel.load_state_dict(t.load("+model.pt"))
    lmodel.to(dev)    # works in place for model
    print(f"y^^ = {lmodel(t.tensor([1.5]).to(dev)).item()}")


