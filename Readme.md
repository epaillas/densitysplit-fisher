## Constraining $\nu \Lambda$CDM using density-split clustering

This repository contains the source code to generate all figures in the paper.

Requirements:
- `densitysplit` (https://github.com/epaillas/densitysplit)

Notes:

- The data that is needed to generate the figures is *not* downloaded automatically when you clone the repository, but it is rather stored in the *Git Large File Storage* cloud. You can download it by running `git lfs pull`, which will convert the pointers that are located in the `data` directory into actual data files. If you do not have Git LFS in your system, you can install it with the `git lfs install` command.

- Figures that are included in the manuscript are stored under `paper_figures/`. There are also support figures for talks/discussions that are stored under `support_figures/`.
