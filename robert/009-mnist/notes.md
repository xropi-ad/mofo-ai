# 009 — MNIST classification

## Task done

Classify handwritten digits (28×28 grayscale, 10 classes) with torchvision's MNIST:
`nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))`,
`CrossEntropyLoss`, Adam `lr=1e-3`, `batch_size=64`, 5 epochs on the GPU.
After each epoch: train loss and val accuracy. The checkpoint/resume code from 008 is kept
(break at epoch 3, resume on the second run).

Result:

```
epoch 0  tloss 0.188  vacc 0.945
epoch 1  tloss 0.114  vacc 0.963
epoch 2  tloss 0.084  vacc 0.969
epoch 3  tloss 0.068  vacc 0.973   (after resume)
epoch 4  tloss 0.051  vacc 0.976
```

Train loss still falling, val accuracy still rising: no overfitting yet.
2.4% wrong ≈ 240 of 10,000 val images.

## Quizzes

- Why is an image `[1, 28, 28]`, not `[28, 28]`? → torch images are `[channels, height, width]`;
  RGB is `[3, 28, 28]`, grayscale `[1, 28, 28]`. Same layout for every image.
- Shape of a batch of 64? → `[64, 1, 28, 28]` (first answer `[64, 784]` was the shape
  `nn.Linear` needs). `in_features = 784`, one input per pixel.
- Flatten with `reshape(64, -1)`? → breaks on the last batch (60,000 % 64 = 32 images):
  `[32, 1, 28, 28]` becomes `[64, 392]` without error. Use `xb.reshape(xb.shape[0], -1)`,
  or the layer `nn.Flatten()`.
- 10 logits, which digit? → the one with the biggest score; for a batch `logits.argmax(dim=1)`.
- `argmax(dim=0)` instead? → no error: shape `[10]`, for each digit the index of the image
  (0..63) with the highest score. A silent bug. `dim=` names the dimension that disappears.
- `CrossEntropyLoss` = `-ln(probability of the right class)`. p = 1 → 0; p = 0.01 → 4.6
  (first answer 2 was log base 10; torch uses ln). Confident mistakes cost a lot.
- Is 5 epochs enough? → yes: 60,000 / 64 ≈ 938 updates per epoch, about 4,700 in total.

## Pitfalls hit

- eval under `i % 200 == 0` (left from 006) ran only at epoch 0.
- `pred.argmax(dim=1)` and `equ.to(int)` without assignment: results thrown away.
  Then `yyb == pred` compared `[64]` with `[64, 10]`. Broadcasting from the right: 64 vs 10 → error.
  (`.sum()` on bools counts the `True` values, so `.to(int)` is not needed.)
- an old `+chk.pt` with `epoch = 10` would make `range(10, 5)` empty: nothing trains, no error.
- `[[1.5]]` prediction left over from the curve fit: not an image.
