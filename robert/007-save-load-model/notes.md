# 007 — saving and loading a model

## Task done

`x.py` trains the 006 model and saves it with `t.save(model.state_dict(), "+model.pt")`.
`x-eval.py` runs in a fresh process: builds the same model, loads the file, predicts `x = 1.5`.
The two predictions match.

## Quizzes

- Keys and shapes of the `state_dict()` for
  `Sequential(Linear(1,256), ReLU(), Linear(256,256), ReLU(), Linear(256,1))`?
  → a flat dict `"<position>.<name>"`, weight shape is `[out, in]`:
  - `0.weight [256, 1]`, `0.bias [256]`
  - `2.weight [256, 256]`, `2.bias [256]`
  - `4.weight [1, 256]`, `4.bias [1]`
- Why is ReLU not in the dict? How do we know it was not a sigmoid? → we do not.
  ReLU has no parameters; it is part of the structure, which lives in code. Loading into a
  model with a sigmoid succeeds silently. The load checks only names and shapes.
  So keep the model definition in one shared file, maybe save a small config next to it.
- A model saved from cuda loads back onto cuda. Load with `t.load(path, map_location=dev)`.
- Why must the input be `[[1.5]]`? → `nn.Linear` works on the **last** dimension, which must
  have size `in_features`. `[1]` works too; `[]` does not; `[3]` means 3 features, not 3 samples.

## Leftovers

- the model definition is copied in both scripts (no shared `make_model()`)
- `x-eval.py` builds data it never uses
- `"+model.pt"` is relative to the current directory
