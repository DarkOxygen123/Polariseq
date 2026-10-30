# Memory and planning

Single-cell datasets now reach millions of cells, more than the memory of the computer most analyses
run on. Polariseq works within a stated amount of memory, and says before reading the data whether an
analysis will fit.

## The budget

The budget is the memory and the number of threads Polariseq may use. Left alone, it claims three
quarters of the memory available when the analysis starts, leaving the rest for the interpreter, the
operating system and whatever else is running: running out of memory makes a computer swap, which is
far slower than an analysis that fits. `ps.set_budget` sets it exactly.

## The plan

`ps.plan` predicts, from a file's description alone (its numbers of cells, genes and stored values), the
peak memory and disk of every step under the budget, and names the step that needs the most. Reading
only the description makes the prediction instant at any size.

When a file is opened, the same prediction picks where the analysis runs. It stays in memory when the
matrix takes at most half the budget. Otherwise it runs out of core, provided the budget covers that
mode's own minimum. Out of core, the blocks each step reads are sized from the budget, so memory
stays within it whatever the size of the dataset. `ps.max_cells` gives the largest dataset a budget can
analyze in each mode.

See also: [Set the memory and threads](../how-to/memory-and-threads.md),
[`ps.plan`](../api/ps/plan.md), [`ps.max_cells`](../api/ps/max_cells.md).
