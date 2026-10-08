# Third-party notices

Polariseq is its own Rust and Python code. Where a step has to give the same
answer as an established tool, the code for that step follows the tool's
source, and the tool's licence covers that part of Polariseq. This file lists
those tools, what Polariseq follows and where, and reproduces the notices
their licences ask to be kept. Each module that follows another project's code
says so in its own header as well.

Some of the code followed is released under the GNU General Public License
(edgeR, limma, statmod, igraph and harmonypy). Polariseq's public release will
therefore carry a licence compatible with the GNU General Public License,
version 3.

This file covers code that Polariseq follows. The libraries it builds on (Rust
crates and Python packages) are dependencies under their own licences, which
their packages carry.

| Project | Licence | Followed in |
| --- | --- | --- |
| Scrublet | MIT | `crates/polariseq-core/src/doublets.rs` |
| NumPy | BSD-3-Clause | `doublets.rs`, `hierarchy.rs` |
| SciPy | BSD-3-Clause | `doublets.rs`, `hierarchy.rs` |
| scikit-image | BSD-2-Clause and BSD-3-Clause | `doublets.rs` |
| scikit-network | BSD-3-Clause | `hierarchy.rs` |
| Scarf | BSD-3-Clause | `hierarchy.rs` |
| umap-learn | BSD-3-Clause | `umap.rs` |
| PyNNDescent | BSD-2-Clause | `nndescent.rs` |
| Scanpy | BSD-3-Clause | `preprocess.rs`, `pca.rs`, `de.rs`, `umap.rs`, `python/polariseq` |
| edgeR | GPL (>= 2) | `pseudobulk.rs` |
| limma | GPL (>= 2) | `pseudobulk.rs` |
| statmod | GPL-2 or GPL-3 | `pseudobulk.rs` |
| igraph | GPL-2.0-or-later | `leiden.rs` |
| harmonypy | GPL-3.0-or-later | `harmony.rs` |

The Rust modules are in `crates/polariseq-core/src/`.

## Code under permissive licences

### Scrublet

<https://github.com/swolock/scrublet>. MIT License, Copyright (c) 2018 Samuel
Wolock.

`doublets.rs` follows Scrublet's doublet detection (Wolock, Lopez and Klein
2019): gene selection by v-score, the simulated doublets, the doublet score of
each cell from its neighbours, and the automatic threshold between singlets
and doublets.

### NumPy

<https://numpy.org>. BSD-3-Clause, Copyright (c) 2005-2025 NumPy Developers.

`doublets.rs` follows `percentile`, `linspace`, `histogram` and `round`, so
that thresholds match Scrublet's. `hierarchy.rs` follows the pairwise summation
of `numpy.sum` (`pairwise_sum` in `loops_utils.h`), so that Paris merges in the
same order as scikit-network.

### SciPy

<https://scipy.org>. BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. and
2003 onwards SciPy Developers.

`doublets.rs` follows `optimize.fmin`, the Nelder-Mead simplex search Scrublet
uses to fit the noise model behind its v-scores. `hierarchy.rs` follows the
layout of `cluster.hierarchy.linkage` and the row sums of `csr_matvec`.

### scikit-image

<https://scikit-image.org>. `doublets.rs` follows `filters.threshold_minimum`,
which Scrublet uses to place its threshold. It lives in
`skimage/filters/thresholding.py`, which scikit-image releases under the
BSD-2-Clause licence (Copyright 2009-2015 Board of Regents of the University of
Wisconsin-Madison, Broad Institute of MIT and Harvard, and Max Planck Institute
of Molecular Cell Biology and Genetics, 2009 Zachary Pincus, 2009 Almar Klein).
The rest of scikit-image is BSD-3-Clause, Copyright 2009-2022 the scikit-image
team.

### scikit-network

<https://github.com/sknetwork-team/scikit-network>. BSD-3-Clause, Copyright (c)
2018 Scikit-network Developers.

`hierarchy.rs` reproduces `Paris` hierarchical clustering (Bonald, Charpentier,
Galland and Hollocou 2018), including its arithmetic, so that the dendrogram
is the same.

### Scarf

<https://github.com/parashardhapola/scarf>. BSD-3-Clause, Copyright (c) 2026
Nygen Analytics AB.

`hierarchy.rs` reproduces `BalancedCut`, Scarf's cut of the Paris dendrogram
into clusters of balanced size (Dhapola et al. 2022).
`preprocess::hvg_scarf` reimplements Scarf's highly variable gene selection
(`mark_hvgs` with `fit_lowess` in `scarf/feat_utils.py`; Scarf 0.32.3, BSD
3-Clause License, Copyright (c) 2021, Dhapola P, Karlsson G): the
minimum-variance trend per mean bin fitted by LOWESS, the corrected variance
and the detection floor of 1% of cells.

### umap-learn

<https://github.com/lmcinnes/umap>. BSD-3-Clause, Copyright (c) 2017 Leland
McInnes.

`umap.rs` follows the fuzzy simplicial set (`smooth_knn_dist`,
`compute_membership_strengths` and the fuzzy union), the curve parameters
(`find_ab_params`) and the layout optimisation with its edge-sampling schedule
(`optimize_layout_euclidean`), including its parallel mode.

### PyNNDescent

<https://github.com/lmcinnes/pynndescent>. BSD-2-Clause, Copyright (c) 2018
Leland McInnes.

`nndescent.rs` builds the approximate neighbour graph the way PyNNDescent
does: a forest of random-projection trees to start, then NN-descent's local
join over new and old candidates, including reverse neighbours.

### Scanpy

<https://github.com/scverse/scanpy>. BSD-3-Clause, Copyright (c) 2025 scverse,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab.

Polariseq gives Scanpy's results for Scanpy's standard steps, and follows its
code where the answer depends on it: quality-control metrics, total-count
normalisation and the `flavor="seurat"` selection of highly variable genes
(`preprocess.rs`), the conventions of `pp.pca` and `pp.scale` (`pca.rs`),
the connectivities of `pp.neighbors` (`umap.rs`), and the t-test of
`tl.rank_genes_groups` with its fold changes (`de.rs` and
`python/polariseq/__init__.py`). The Python package stores results in the
places Scanpy uses, so that Scanpy's tools and plots read them.

## Code under the GNU General Public License

### edgeR

<https://bioconductor.org/packages/edgeR>. GPL (>= 2), Copyright Gordon
Smyth, Yunshun Chen, Aaron Lun, Davis McCarthy, Mark Robinson and
contributors.

`pseudobulk.rs` follows `filterByExpr`, `.calcFactorTMM` and `normLibSizes`
(edgeR 4.10) for the comparison of conditions on pseudobulk counts.

### limma

<https://bioconductor.org/packages/limma>. GPL (>= 2), Copyright Gordon Smyth
and contributors.

`pseudobulk.rs` follows `lmFit`, `fitFDist`, `trigammaInverse`, `squeezeVar`
and `eBayes` with `trend = TRUE` (limma 3.68).

### statmod

<https://cran.r-project.org/package=statmod>. GPL-2 or GPL-3, Copyright Gordon
Smyth and contributors.

`pseudobulk.rs` follows `logmdigamma`.

### igraph

<https://igraph.org>. GPL-2.0-or-later, Copyright (C) 2020-2025 The igraph
development team.

`leiden.rs` structures the Leiden algorithm (Traag, Waltman and van Eck 2019)
as igraph's `igraph_community_leiden` does (`src/community/leiden.c`): fast
local moving from a queue of unstable nodes, the refinement into
well-connected parts, and two iterations from the previous partition, as
`community_leiden(n_iterations=2)` runs.

### harmonypy

<https://github.com/slowkow/harmonypy>. GPL-3.0-or-later, Copyright (C) 2018
Ilya Korsunsky, 2019 Kamil Slowikowski.

`harmony.rs` follows harmonypy 2.0.2's C++ backend (`src/harmony.cpp`), a
step-by-step port of R harmony 2 (<https://github.com/immunogenomics/harmony>,
GPL-3): the initial centroids, the soft assignments with the diversity penalty
updated in shuffled blocks, the objective and its convergence rules, and the
ridge correction per cluster, with the penalty estimated from the expected
counts and several batch variables at once (Korsunsky et al. 2019).

The GNU General Public License, version 3, is at
<https://www.gnu.org/licenses/gpl-3.0.html>, and version 2 at
<https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.

## Licence texts

### Scrublet licence

```text
The MIT License (MIT)
Copyright (c) 2018 Samuel Wolock

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### NumPy licence

```text
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### SciPy licence

```text
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### scikit-image licence

The entries of scikit-image's licence file that cover the code followed, and
the licence texts they name.

```text
Files: *
Copyright: 2009-2022 the scikit-image team
License: BSD-3-Clause

Files: skimage/filters/thresholding.py
       skimage/graph/_mcp.pyx
       skimage/graph/heap.pyx
Copyright: 2009-2015 Board of Regents of the University of
           Wisconsin-Madison, Broad Institute of MIT and Harvard,
           and Max Planck Institute of Molecular Cell Biology and
           Genetics
           2009 Zachary Pincus
           2009 Almar Klein
License: BSD-2-Clause

License: BSD-2-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

License: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### scikit-network licence

```text
BSD License

Copyright (c) 2018, Scikit-network Developers
Bertrand Charpentier <bertrand.charpentier@live.fr>
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Scarf licence

```text
BSD 3-Clause License

Copyright (c) 2026, Nygen Analytics AB
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### umap-learn licence

```text
BSD 3-Clause License

Copyright (c) 2017, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### PyNNDescent licence

```text
BSD 2-Clause License

Copyright (c) 2018, Leland McInnes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Scanpy licence

```text
BSD 3-Clause License

Copyright (c) 2025 scverse®
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
