#! /opt/venv/bin/python3

import torch as t


x = t.tensor([0., 1, 2, 3])
y = 2*x + 1

w = t.tensor(0., requires_grad=True)
b = t.tensor(0., requires_grad=True)

# pred = w*x + b
# loss = ((pred - y) ** 2).mean()
# loss.backward()
# print(w.grad, b.grad)

lr = 0.2
for i in range(100):
    pred = w*x + b
    loss = ((pred - y)**2).mean()
    loss.backward()
    with t.no_grad():
        w -= w.grad*lr
        b -= b.grad*lr
    #print(w.grad, b.grad)
    #print(f"loss = {loss}   w:{w}  b:{b}")
    w.grad = None
    b.grad = None
print(f"loss = {loss}   w:{w}  b:{b}")


