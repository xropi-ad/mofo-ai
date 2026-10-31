# 002 — the same loop, the torch way

## Task done

Rewrite 001 with torch parts:

- `w`, `b` → `model = nn.Linear(1, 1)`
- `w * x + b` → `model(x)`
- hand-written MSE → `nn.MSELoss()`
- update under `no_grad()` → `opt.step()` with `optim.SGD(model.parameters(), lr=0.1)`
- `w.grad = None` → `opt.zero_grad()`

## Quizzes

- Input shape for `nn.Linear`? → `[batch, features]`. `x` of shape `[4]` must become `[4, 1]`.
- Why does `print(model.parameters)` show no numbers? → without `()` it is the method itself;
  with `()` it is a generator. Use `model.weight.item()`, `model.bias.item()`.
- `nn.Linear` starts with random values. What if you run the script twice? → slightly different
  results; same minimum, different path. Fix with `t.manual_seed(0)` before creating the model.
  Use the same seed when comparing two settings.
