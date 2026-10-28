#!/usr/bin/env python3
import torch, torch.nn as nn
import torchvision as tv
import torchvision.transforms as T

dev = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/mount/data/cifar10"

norm = T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
test_tf = T.Compose([T.ToTensor(), norm])

train_ds = tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
test_ds = tv.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=512, num_workers=4)

# ResNet-18, adapted to 32x32 images: smaller first conv, no max-pool
model = tv.models.resnet18(num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model = model.to(dev)

epochs = 5
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=epochs, steps_per_epoch=len(train_dl))
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for x, y in train_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(dev), y.to(dev)
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch + 1}: loss {loss.item():.3f}, test acc {correct / len(test_ds):.3f}")

torch.save(model.state_dict(), "/mount/data/resnet18-cifar10.pt")

