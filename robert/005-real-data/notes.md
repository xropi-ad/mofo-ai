# 005 — training in batches

## Task done

Train the 004 network with `TensorDataset` + `DataLoader(batch_size=10, shuffle=True)`:
an outer loop over epochs, an inner loop over batches. Data stays on the CPU; each batch
moves to the GPU inside the loop.

## Quizzes

- 50 samples, `batch_size=10`: how many batches per epoch? → 5.
- Shape of `xb`? → `[10, 1]`: batch size, then the shape of one sample.
- The old loop made 2000 updates. How many epochs give the same? → 2000 / 5 = 400.
- The printed `loss` is now only the **last batch**. How to get the loss on all data?
  → recompute it once after training, under `no_grad()`.
- What if the dataset does not fit in memory? → write your own `Dataset` that reads one sample
  from disk in `__getitem__`. `DataLoader(num_workers=...)` reads batches in parallel.
- Full loss batch by batch: why divide by the number of **samples**, not batches?
  → with 50 samples and `batch_size=16` the batches are 16, 16, 16, 2.
  Correct: `(BL1+BL2+BL3)*16/50 + BL4*2/50` — each sample counts 1/50.
  The naive `(BL1+…+BL4)/4` gives each sample in the last batch 1/8, the others 1/64.
  In code: `total += loss.item() * len(xb)`, then `total / len(ds)`.

## Pitfalls hit

- `xb.to(dev)` without assignment — the same bug as in 003. Check with `print(xb.device)`.
- `.clone()` exists only on tensors; a model copy is `copy.deepcopy(model)`. But a copy made
  before training stays untrained. To evaluate on the CPU, move the trained model itself.
