#! /opt/venv/bin/python3

import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

t.manual_seed(0)
dev = "cuda" if t.cuda.is_available() else "cpu"


tds = datasets.MNIST("+data", train=True, download=True, transform = transforms.ToTensor())
vds = datasets.MNIST("+data", train=False, download=True, transform = transforms.ToTensor())

tdl = DataLoader(tds, batch_size=64, shuffle=True)
vdl = DataLoader(vds, batch_size=64, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(13*13*16, 10)
)
#opt = t.optim.SGD(model.parameters(), lr=0.001)
opt = t.optim.Adam(model.parameters(), lr=0.001)


model.to(dev)


for e in range(5):
    for x, y in tdl:
        xdev = x.to(dev)
        ydev = y.to(dev)
        pred = model(xdev)
        loss = loss_fn(pred, ydev)
        loss.backward()
        opt.step()
        opt.zero_grad()
    with t.no_grad():
        sum_loss = 0
        n = 0
        for x, y in vdl:
            xdev = x.to(dev)
            ydev = y.to(dev)
            pred = model(xdev)
            sum_loss += loss_fn(pred, ydev)
            n += 1
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

        print(f"loss = {sum_loss/n} vaccuracy = {acc_equ/acc_tot}")





