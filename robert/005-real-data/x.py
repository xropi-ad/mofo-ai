#! /opt/venv/bin/python3

import torch as t
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

t.manual_seed(0)    # to make it deterministic
dev = "cuda" if t.cuda.is_available() else "cpu"

    


# One trap: nn.Linear expects input shape [batch, features]. Your x has shape [4]. It needs shape [4, 1]: 4 samples, 1 feature each. The same holds for y.
x = t.linspace(-2, 2, 50).reshape(50, 1)
y = x**2

model = nn.Sequential(  nn.Linear(1, 16),    # 1 in => 16 hidden
                        nn.ReLU(),
                        nn.Linear(16, 1)
)
#w = t.tensor(0., requires_grad=True)
#b = t.tensor(0., requires_grad=True)

loss_fn = nn.MSELoss()

x = x.to(dev)
y = y.to(dev)
#model = model.to(dev)
model.to(dev)    # works in place for model

ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=10, shuffle=True)

opt = t.optim.SGD(model.parameters(), lr=0.1)

for i in range(400):
    for xb, yb in dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

#print(f"loss = {loss}   model.parameters: {model.weight.item()} {model.bias.item()}")
print(f"loss = {loss}")
with t.no_grad():
    print(f"y^ = {model(t.tensor([[1.5]]).to(dev)).item()}")


