# 003 — training on the GPU

## Task done

Run the 002 loop on cuda: `dev = "cuda" if t.cuda.is_available() else "cpu"`, then move
`x`, `y` and `model` to `dev`.

## Quizzes

- Which objects must be on the GPU? → `x`, `y` and `model`.
- Why did `x.to(dev)` not work? → for a tensor, `.to()` returns a **new** tensor:
  `x = x.to(dev)`. For a model, `.to()` moves its parameters **in place**: `model.to(dev)`.
- Why does the optimizer still work when the model is moved after creating it? → the optimizer
  holds references to the same parameter objects. `model.to()` swaps the data inside each
  object; the objects stay the same. (Usual order anyway: move the model, then create `opt`.)
- Why do results look like doubles? → the tensors are float32; `.item()` returns a Python
  `float`, which is always 64-bit, so you see more digits than the value really has.
