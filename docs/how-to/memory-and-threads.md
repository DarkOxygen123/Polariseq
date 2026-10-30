# Set the memory and threads

```python
import polariseq as ps

ps.set_budget(ram_gb=6, threads=4)
print(ps.budget())       # the budget in force
print(ps.resources())    # what the machine has
```

Set the budget before the first step. Anything you leave out is detected: the memory budget is three
quarters of the memory available at that moment, and the threads are the machine's performance cores.

## Choose the values

- **`ram_gb`**: what Polariseq may hold in memory. Leave room for the rest of your session (a notebook,
  a browser). On a cluster node, use the memory the job was given.
- **`threads`**: the worker threads of every parallel step. They are set once per Python session; a
  later call with another number has no effect and warns.
- **`spill_dir`, `spill_gb`**: where out-of-core scratch files go, and how much space they may use.

## Pin the budget for reproducible runs

Without `ram_gb`, the budget follows the memory free when the analysis starts, and the block size of
out-of-core passes follows the budget. Pin both values so two runs of the same script match.

See also: [`ps.set_budget`](../api/ps/set_budget.md), [`ps.resources`](../api/ps/resources.md),
[Memory and planning](../background/memory.md).
