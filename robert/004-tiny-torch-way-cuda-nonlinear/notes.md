# 004 — a nonlinear network

## Task done

Fit the curve `y = x²` (50 points in `[-2, 2]`) with
`nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))`, 2000 steps.
Prediction for `x = 1.5`: 2.33 (true value 2.25).

## Quizzes

- Can a stack of `nn.Linear` layers with nothing between them fit `x²`? → no:
  `w2*(w1*x + b1) + b2` is still `a*x + c`. Linear on linear is linear.
- The fix: a nonlinear function between layers, e.g. ReLU `max(0, z)`. Each ReLU adds a bend;
  the network builds the curve from short straight pieces. The hidden size (16) is how many.
- How to predict for `x = 1.5`? → the input needs the training shape and the model's device:
  `model(t.tensor([[1.5]]).to(dev))`.
- Wrap prediction in `with t.no_grad():` — no `.backward()` follows, so no graph is needed.
