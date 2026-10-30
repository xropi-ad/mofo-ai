#! /opt/venv/bin/python3

import torch as t
from torch import nn

t.manual_seed(0)    # to make it deterministic

# One trap: nn.Linear expects input shape [batch, features]. Your x has shape [4]. It needs shape [4, 1]: 4 samples, 1 feature each. The same holds for y.
x = t.tensor([[0.], [1], [2], [3]])
y = 2*x + 1

model = nn.Linear(1, 1)
opt = t.optim.SGD(model.parameters(), lr=0.1)
#w = t.tensor(0., requires_grad=True)
#b = t.tensor(0., requires_grad=True)

loss_fn = nn.MSELoss()

for i in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
print(f"loss = {loss}   model.parameters: {model.weight.item()} {model.bias.item()}")


