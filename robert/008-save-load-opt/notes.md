# 008 — resuming training (checkpoint)

## Task done

`x.py` trains the 006 model. At epoch 1500 it saves a checkpoint and exits, if none exists yet.
On the next start it finds `+chk.pt`, loads it, and continues from epoch 1500 to 3000.

The checkpoint is one dict:

```python
t.save({"opt": opt.state_dict(), "model": model.state_dict(), "epoch": i}, "+chk.pt")
```

## Quizzes

- Why save the optimizer too? → Adam keeps its own tensors per parameter: running averages of
  past gradients and of their squares. They set the step sizes. A fresh Adam starts from zeros.
- Where is the epoch stored? → in neither `state_dict()`. Add it to the checkpoint dict as a
  plain number; any Python value can go there.
- Resume from `epoch` or `epoch + 1`? → depends on when you save. Here the save happens at the
  start of epoch 1500, before training it, so resume at 1500.
- Order of the resume lines? → `model.to(dev)` → create `opt` → `opt.load_state_dict()`.
  `opt` puts the loaded tensors on the device of the parameters it holds at that moment.
- Load with `t.load(path, map_location=dev)`.

- Run A trains 3000 epochs without a break, run B breaks and resumes. Same losses after 1500?
  First answer: "yes, `t.manual_seed(0)` is set." → no. The seed makes the **start** of each
  process the same (same noise, same split). But each random call takes the next numbers from
  the generator. At epoch 1500, run A's shuffle uses the numbers after 1500 epochs of shuffles;
  run B's uses the numbers run A used at epoch 0. So the batch order differs.
  Fix: also save `t.get_rng_state()`, restore with `t.set_rng_state(chkd["rng"].cpu())`
  (`map_location` would move it to cuda; it must stay on the CPU).
  In practice an exact repeat is rarely needed, and some GPU ops are not exactly repeatable.
