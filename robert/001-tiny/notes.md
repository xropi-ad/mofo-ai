# 001 — tiny training loop by hand

## Task done

Fit a line to data from `y = 2x + 1` with plain tensors only: parameters `w`, `b` with
`requires_grad=True`, forward, MSE loss, `backward()`, gradient descent update, gradient reset.

## Tensor basics (quizzes before this exercise)

- **Shape** of `[[[1,2],[3,4],[5,6]], [[7,8],[9,0],[1,2]]]`? → `[2, 3, 2]`. Read from the outside in.
- **Indexing**: each index removes one dimension from the front.
  `c[1]` → shape `[3, 2]`; `c[1, 0]` → shape `[2]`, value `[7, 8]`.
- **Slicing**: `:` keeps the dimension. `c[:, 0, 1]` → shape `[2]`, value `[2, 8]`.
- **dtype**: `x.mean()` fails on int64. Fix: `x.to(torch.float32)`, `x.float()`, or `dtype=` at creation.
- **`*` vs `@`**: `*` is element-wise. `@` is matrix multiply: `[n, k] @ [k, m]` → `[n, m]`.
  `[[1,0,2],[0,1,1]] @ [[1,1],[0,1],[1,0]]` → `[[3, 1], [1, 1]]`.
- **reshape**: same numbers, same order, new shape; the count must match.
  `[6].reshape(4, 2)` → error; `reshape(-1, 1, 2)` → `[3, 1, 2]`.
- **Broadcasting**: compare shapes from the right; sizes fit if equal or one is 1.
  `[2, 3] + [2, 1]` → `[2, 3]`; `[2, 3] + [2]` → error (3 vs 2).
- **Device**: a cuda tensor plus a cpu tensor → error. Torch never moves data for you.
- **Autograd**: forward only records a graph; `.backward()` computes and stores `.grad`,
  only on tensors created with `requires_grad=True`.
  `y = 5x + 1`, `x = 2` → `x.grad = 5` (the derivative does not depend on `x`).

## Quizzes in this exercise

- `w.grad = -17`: should `w` go up or down? → up. Update: `w = w - lr * w.grad` (gradient descent).
- Why does `w = w - lr * w.grad` fail in torch? → it makes a new, recorded tensor; the new `w` is an
  in-between result and has no `.grad` next round. Fix: `with torch.no_grad(): w -= lr * w.grad`.
- Why is `no_grad()` needed? → in-place change of a leaf tensor with `requires_grad=True` raises
  `RuntimeError`. The update is not part of the model, so it must not be recorded.
- What happens to `w.grad` in round 2? → `.backward()` **adds** to `.grad`. Reset it: `w.grad = None`.
- A `backward()` before the loop leaves gradients behind, which the first loop step adds to.
- After 100 steps, is `w` exactly 2? → no; steps get smaller as the loss gets smaller.
- Which two knobs control how close you get? → steps (more = closer) and lr (bigger = faster).
- lr zones: small = slow; bigger but below a threshold = oscillates, then settles;
  above the threshold (about 0.24 here) = blows up to `inf` / `nan`.
  Checked: `lr = 0.1` converges, `lr = 0.3` blows up.
  With noisy batches (real training), a big lr also keeps the result jumping near the minimum.

## Pitfalls hit

- Typo `pred- - y` computed `pred + y`.
