# 010 — first CNN on MNIST

## Task done

Replace the 009 MLP with a small CNN. Same data, Adam `lr=0.001`, `batch_size=64`, 5 epochs:

```python
nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(13*13*16, 10),
)
```

Result after 5 epochs:

```
009 MLP            val accuracy 0.976   101,770 parameters
010 CNN + Adam     val accuracy 0.981    27,210 parameters   val loss 0.058
010 CNN + SGD      val accuracy 0.894   (lr=0.001, optimizer not comparable with 009)
```

Wrong images fell from about 240 to about 190 of 10,000, with about 3.7 times fewer parameters.
This is one run with one seed, so the exact gap is not precise.

Predictions before the run: "better" and "fewer". Both were right.

## Quizzes

- Shuffle the 784 pixel positions with one fixed rule, for all images. Train the 009 MLP from
  scratch. Val accuracy? → about the same. Fully connected means all pixel pairs are treated the
  same, so the MLP does not use the 2D structure. The images look like noise to a human.
- A neuron learned a stroke in the top-left. Does it fire on the same stroke in the bottom-right?
  → no. Its weights there do almost nothing. An MLP must learn the pattern again for each position.
- How to find a stroke anywhere with only 9 weights? → slide a 3×3 kernel over the image. At each
  position: multiply with the pixels under it and add up. The output is a feature map. All
  positions use the same weights (**weight sharing**).
- Feature map size for 28×28, 3×3 kernel, no padding? → 28 − 3 + 1 = 26.
- Shape after `Conv2d(1, 16, 3)` for `[64, 1, 28, 28]`? → `[64, 16, 26, 26]` (first answer
  `[64, 1, 26, 26]`). Each kernel gives its own map, so channels = number of kernels.
- Parameters of `Conv2d(1, 16, 3)`? → (9 + 1) × 16 = 160. First answer 26×26×9×16 is the number of
  multiplications per image. Positions add compute, not parameters.
- `in_features` of the last `Linear`? → `[64, 16, 26, 26]` → pool → `[64, 16, 13, 13]` → flatten
  → `[64, 2704]`, so 13 × 13 × 16 = 2704.
- Parameters of `Linear(2704, 10)`? → 2704 × 10 + 10 = 27,050. First answers 3k and 160 were wrong:
  2704 is the number of inputs, and 160 is the conv layer. More than 99% of the CNN's parameters
  are in this last layer.

## Terms

- **Kernel** (filter): the small set of weights that slides over the image. Trained by `backward()`.
- **Feature map**: the output of one kernel; how strongly its pattern appears at each position.
- **Channels**: planes per image. Colors at the input, one per kernel after a conv layer.
- **`MaxPool2d(2)`**: keeps the largest number of each 2×2 block. Halves height and width, has no
  parameters, and makes small shifts matter less.

## Pitfalls hit

- `from torch import DataLoader` → it is in `torch.utils.data`.
- `t.cuda_available()` → `t.cuda.is_available()`.
- `SGD` with `lr=0.001` instead of Adam: 0.894. Model and optimizer both changed, so the run said
  nothing about CNNs. Change one thing at a time.

## Leftovers

- `sum_loss / n` divides by batches; the last val batch has only 16 images. Use
  `loss * len(x)` and divide by `len(vds)`, as in 005 and 006.
- the printed loss is val loss; 009 printed train loss, so the two are not comparable
- no parameter count in the script; the totals above were computed by hand
