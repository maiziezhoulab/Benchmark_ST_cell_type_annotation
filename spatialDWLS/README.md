# spatialDWLS

## Overview
This folder contains a benchmark-oriented `spatialDWLS` workflow for leave-one-section-out cell type annotation on the MERFISH spinal cord dataset. The implementation is more elaborate than a minimal package example because it must export data from `h5ad`, construct a reference incrementally, and remain numerically stable across many sweep configurations.

The workflow proceeds in three stages:
1. export the held-out section as cell-level counts and coordinates;
2. export a reference matrix and cell-type labels from an increasing set of training sections;
3. run DWLS deconvolution, with a guarded fallback to robust NNLS/WLS if the package solver fails.

## Files
- `run_dwls_sweep.sh`: orchestrates the ordered reference-size sweep for one held-out section.
- `export_cells_no_binning.py`: exports the test section as one cell per spot, without binning.
- `export_reference_from_h5ad.py`: exports the reference expression matrix and cell-type labels from one or more training sections.
- `run_dwls.R`: performs preprocessing, signature construction, deconvolution, diagnostics, and output writing.

## Inputs
The workflow assumes a MERFISH `.h5ad` with at least:
- `obs['Section ID']`
- `obs['MERFISH cell type annotation']`
- spatial coordinates in `obsm['spatial']` by default

Default configuration in `run_dwls_sweep.sh`:
```bash
H5AD=/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad
LABEL_COL="MERFISH cell type annotation"
SPATIAL_KEY="spatial"
PY_ENV=clustering
R_ENV=giotto-r44
```

The wrapper can use either `conda` or `micromamba`, and attempts to detect the appropriate manager automatically.

## Usage
Run the full sweep for one held-out section:
```bash
./run_dwls_sweep.sh 0503_M4_S
```

For each reference size `i`, the script creates:
- a reference directory:
```text
ref_train_<TEST>__ref<i>of17__<section1+section2+...>
```
- a result directory:
```text
results_<TEST>__ref<i>of17__<section1+section2+...>
```

It also creates one test-export directory per held-out section:
```text
work_cells_<TEST>
```

## Intermediate exports
`export_cells_no_binning.py` writes:
- `giotto_counts.txt`: genes × spots count matrix, where each spot is one original cell;
- `giotto_coords.txt`: `spot`, `sdimx`, `sdimy`;
- `true_props.csv`: one-hot ground-truth cell-type proportions.

`export_reference_from_h5ad.py` writes:
- `ref_sc_matrix.txt`: genes × cells reference matrix;
- `ref_celltypes.txt`: two-column `cell_id` / `cell_type` mapping.

## Main processing choices
`run_dwls.R` does more than a naive DWLS call. It:
- compares `CPM` versus `log1p(CPM)` and selects the transform with higher cross-cell-type variance;
- keeps up to 1500 shared high-variance genes;
- builds a mean reference signature per cell type;
- prunes zero-variance, duplicate, and collinear cell-type signatures;
- attempts `DWLS::solveDampenedWLS` first;
- falls back to a robust NNLS/WLS solver if DWLS is unavailable or unstable.

This defensive design is methodologically sound for benchmarking because it reduces avoidable numerical failures. However, it also means the implementation is no longer a pure package-default baseline. Any manuscript or comparison table should state that this is a stabilized benchmark wrapper around DWLS rather than an untouched reference run.

## Outputs
Each result directory contains:
- `dwls_props.csv`: spot-by-cell-type proportions.
- `dwls_toptype.tsv`: top predicted cell type and its proportion for each spot.
- `dwls_toptype.png`: spatial visualization of top predicted labels, if plotting dependencies are available.

## Evaluation interpretation
The exported `dwls_toptype.tsv` is the natural hard-label output for downstream scoring. Because each MERFISH cell is treated as one spot, this benchmark effectively evaluates whether the dominant inferred type matches the original annotation.

That framing is coherent for this dataset, but it compresses a deconvolution method into a classifier. The proportions in `dwls_props.csv` remain the more informative primary output.

## Dependencies
Python side:
- `scanpy`
- `numpy`
- `pandas`

R side:
- `data.table`
- `DWLS`
- `nnls`
- optionally `ggplot2` and `viridis` for plotting

## Notes
- The folder name is `spatialDWLS_new`; this README documents the current stabilized benchmark implementation in that directory.
- The wrapper assumes the held-out target section is cell-resolved and therefore does not perform spatial binning.
- Automatic fallback to NNLS/WLS improves robustness, but it also changes the effective solver when DWLS fails.
