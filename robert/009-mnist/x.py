#! /opt/venv/bin/python3

true = True
false = False

import torch as t
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from torchvision import datasets, transforms

t.manual_seed(0)    # to make it deterministic
dev = "cuda" if t.cuda.is_available() else "cpu"

tds = datasets.MNIST("+data", train=true, download=true, transform=transforms.ToTensor())
vds = datasets.MNIST("+data", train=false, download=true, transform=transforms.ToTensor())

model = nn.Sequential(    nn.Flatten(),
                          nn.Linear(28*28, 128),
                          nn.ReLU(),
                          nn.Linear(128, 10)
)

loss_fn = nn.CrossEntropyLoss()

model.to(dev)    # works in place for model

tdl = DataLoader(tds, batch_size=64, shuffle=True)
vdl = DataLoader(vds, batch_size=64, shuffle=True)

#opt = t.optim.SGD(model.parameters(), lr=0.1)
opt = t.optim.Adam(model.parameters(), lr=0.001)

if Path("+chk.pt").exists():
    chkd = t.load("+chk.pt", map_location=dev)
    opt.load_state_dict(chkd["opt"])
    model.load_state_dict(chkd["model"])
    epoch_start = chkd["epoch"]
else:
    epoch_start = 0

for i in range(epoch_start, 5):
    print(i)
    if i == 3:
        if not Path("+chk.pt").exists():
            print("break")
            t.save({
                "opt": opt.state_dict(),
                "model": model.state_dict(),
                "epoch": i
            }, "+chk.pt")
            exit(0)

    for xb, yb in tdl:
        xdb = xb.to(dev)
        ydb = yb.to(dev)
        pred = model(xdb)
        loss = loss_fn(pred, ydb)
        loss.backward()
        opt.step()
        opt.zero_grad()
    if i%1 == 0:
        with t.no_grad():
            totalt = 0
            for xxb, yyb in tdl:
                xxb = xxb.to(dev)
                yyb = yyb.to(dev)
                pred = model(xxb)
                loss = loss_fn(pred, yyb)
                totalt += loss*len(xxb)
            acc_equ = 0
            acc_tot = 0
            for xxb, yyb in vdl:
                xxb = xxb.to(dev)
                yyb = yyb.to(dev)
                pred = model(xxb)    # [64, 10]
                pred = pred.argmax(dim=1)
                equ = yyb == pred
                equ = equ.to(int)
                acc_equ += equ.sum()
                acc_tot += len(equ)
            print(f"{i/1} tloss = {totalt / len(tds)}   vaccuracy = {acc_equ/acc_tot}")


#with t.no_grad():
#    print(f"y^ = {model(t.tensor([[1.5]]).to(dev)).item()}")

