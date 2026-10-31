#! /opt/venv/bin/python3

import torch as t
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

t.manual_seed(0)    # to make it deterministic
dev = "cuda" if t.cuda.is_available() else "cpu"


# One trap: nn.Linear expects input shape [batch, features]. Your x has shape [4]. It needs shape [4, 1]: 4 samples, 1 feature each. The same holds for y.
x = t.linspace(-2, 2, 20).reshape(20, 1)
#y = x**2
y = x**2 + 0.3 * t.randn_like(x)        # add noise to overfit to


model = nn.Sequential( nn.Linear(1, 256),
                       nn.ReLU(),
                       nn.Linear(256, 256),
                       nn.ReLU(),
                       nn.Linear(256, 1)
)

loss_fn = nn.MSELoss()

model.to(dev)    # works in place for model

ds = TensorDataset(x, y)
tds, vds = random_split(ds, [15, 5])
tdl = DataLoader(tds, batch_size=5, shuffle=True)
vdl = DataLoader(vds, batch_size=20, shuffle=True)
xx, yy = next(iter(vdl))

#opt = t.optim.SGD(model.parameters(), lr=0.1)
opt = t.optim.Adam(model.parameters(), lr=0.01)


for i in range(3000):
    for xb, yb in tdl:
        xdb = xb.to(dev)
        ydb = yb.to(dev)
        pred = model(xdb)
        loss = loss_fn(pred, ydb)
        loss.backward()
        opt.step()
        opt.zero_grad()
    if i%200 == 0:
        with t.no_grad():
            totalv = 0
            for xxb, yyb in vdl:
                xxb = xxb.to(dev)
                yyb = yyb.to(dev)
                pred = model(xxb)
                loss = loss_fn(pred, yyb)
                totalv += loss*len(xxb)
            totalt = 0
            for xxb, yyb in tdl:
                xxb = xxb.to(dev)
                yyb = yyb.to(dev)
                pred = model(xxb)
                loss = loss_fn(pred, yyb)
                totalt += loss*len(xxb)
            print(f"{i/200} tloss = {totalt / len(tds)}   vloss = {totalv / len(vds)}")


with t.no_grad():
    print(f"y^ = {model(t.tensor([[1.5]]).to(dev)).item()}")


