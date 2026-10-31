# 006 — train / validation split

## Task done

Make overfitting visible: 20 noisy points `y = x² + 0.3 * randn`, split 15 train / 5 val with
`random_split`, a big model (two hidden layers of 256), Adam `lr=0.01`, 3000 epochs.
Every 200 epochs print train and val loss, each over its whole set, batch by batch.

Result: train loss fell to about 0.0003; val loss stopped improving early and stayed near 0.06.

## Quizzes

- Train loss down, val loss down → the model generalizes.
- Train loss down, val loss **up** → overfitting.
- Prediction: when does val loss start to rise? → said "late". The gap grew late, but val loss
  stopped improving early. With only 5 val points the rise is weak.
- What MSE does a perfect model get on noisy data? → the mean of `noise²` = 0.3² × 1 = 0.09.
  A train loss far below the noise level means the model memorizes the noise.
- "The noise is fixed, so the model learns `x²` + noise, not `x²`." → true for the train points.
  But each val point has its own, independent noise. Learned train noise does not help there;
  it even bends the curve the wrong way. Train loss = fit to the training set; val loss = fit to
  the task.

## Pitfalls hit

- `model.to('cpu')` inside the eval step moved the model itself (in place), so the next training
  step failed with a device error. Fix: evaluate on the GPU, batch by batch.
- Getting one batch: `next(iter(dl))`. Each `iter()` / `for` starts a fresh pass.
- First run had no noise in `y`, so there was nothing to overfit.
- Divided by `len(vdl)` (number of batches) instead of `len(vds)` (number of samples).
- Small jumps in the losses are normal with Adam and tiny batches; read the trend.
