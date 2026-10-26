---
name: Bug report
about: Something crashed, hung, or produced wrong results
title: "[bug] "
labels: bug
assignees: ""
---

**What happened?**

<!-- A clear description of the bug. If it crashed, paste the FULL traceback. -->

**What did you expect to happen?**

**Minimal code to reproduce**

```python
# import polariseq as ps
# ...
```

**Input data**

<!-- How was the dataset produced? Format (.h5ad / 10x .h5 / .mtx), number of
cells × genes, where it came from (Cell Ranger version, scanpy version, public
dataset ID). If you can share a small file that triggers the bug, link it. -->

**Environment**

```
Output of: python -c "import polariseq as ps; print(ps.get_build_info())"

OS + version:
Python version:
RAM:
Disk type (HDD/SSD):
```

**Additional context**

<!-- Anything else — error messages in a notebook, warnings, screenshots of
wrong-looking plots, comparison with scanpy output if you tried it. -->
