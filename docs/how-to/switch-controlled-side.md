# Switch the controlled side

By default the controlled side is `Side.GUARD`. Train the bandit
instead with one `dataclasses.replace`:

## Recipe

```python
import dataclasses
from orbitalgym import Side, make_lady_bandit_guard

cfg = make_lady_bandit_guard()
cfg = dataclasses.replace(cfg, controlled_side=Side.BANDIT)
```

Now `cfg.guard_policy` drives the guard, and the adapter's
action / observation are the bandit's.

## Effect on adapters

- **Gymnasium** — observation and action space switch to the bandit's.
- **PettingZoo** — multi-agent, both sides controlled; `controlled_side`
  is ignored.
- **POMDP** — per-side dispatch, `controlled_side` is ignored;
  pass `side` explicitly to `reward` / `observation`.

## See also

- [Use the Gymnasium adapter](use-gymnasium-adapter.md).
- [Extending → Controlled-side policy cookbook](../extending/controlled-policy-cookbook.md)
  for patterns once you've switched.
