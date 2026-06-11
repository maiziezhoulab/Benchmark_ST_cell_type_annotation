# cell2location

## Overview
This folder contains a benchmark-specific `cell2location` workflow for leave-one-section-out cell type annotation on the MERFISH spinal cord dataset. The implementation is not a generic tutorial: it is a controlled sweep in which one section is held out as the target and the reference is expanded incrementally from 1 to 17 training sections.

The core logic is split into two stages:
1. Learn cell-type signatures from the training sections with `RegressionModel`.
2. Map the held-out section with `Cell2location`, then score hard labels by argmax over inferred abundances.

## Files
- `run_cell2l_sweep.sh`: orchestrates the ordered reference-size sweep for one held-out section.
- `cell2l_pipeline.py`: main benchmark pipeline, including training, mapping, H5AD export, and metric computation.
- `run_cell2loc_merfish.py`: earlier single-configuration script; useful for inspection, but `cell2l_pipeline.py` is the reproducible entry point used by the sweep.
- `inspect.ipynb`, `plot.ipynb`: exploratory notebooks.

## Inputs
The scripts assume a MERFISH `.h5ad` with at least:
- `obs['Section ID']`
- `obs['MERFISH cell type annotation']`
- a shared gene space across all sections

By default, `run_cell2l_sweep.sh` uses:
```bash
H5AD=/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad
LABEL_COL="MERFISH cell type annotation"
SECTION_COL="Section ID"
```

## Usage
Run the full sweep for one held-out section:
```bash
bash run_cell2l_sweep.sh 0503_M4_S
```

This script:
- removes the test section from the ordered master list of 18 sections;
- builds nested training sets of size `1..17`;
- runs `cell2l_pipeline.py` once per training-set size;
- writes one output directory per run, named as:
```text
c2l_<TEST_SECTION>__ref<i>of17__<section1+section2+...>
```

Example:
```text
c2l_0503_F4_C__ref10of17__0503_F5_T+0503_F5_C+...
```

## Main parameters
Important defaults in `run_cell2l_sweep.sh`:
- `MAX_EPOCHS_REF=50`
- `MAX_EPOCHS_MAP=300`
- `N_CELLS_PER_LOCATION=1.0`
- `DETECTION_ALPHA=50.0`

These choices are reasonable for single-cell-resolution MERFISH, but they remain assumptions. In particular, the hard-coded `N_CELLS_PER_LOCATION=1.0` encodes a strong prior that each target location behaves like one cell. That is coherent for this benchmark, but should not be transferred uncritically to coarser spatial assays.

## Outputs
Each run directory contains:
- `reference_signatures_celltype_by_gene.csv`: inferred reference signatures.
- `mapped_<TEST_SECTION>.h5ad`: mapped target AnnData with posterior outputs.
- `abundances_<TEST_SECTION>_q05_cell_abundance_w_sf.csv` or `abundances_<TEST_SECTION>_means_cell_abundance_w_sf.csv`: inferred cell-type abundance matrix.
- `predictions_<TEST_SECTION>.csv`: per-cell true and predicted labels.
- `confusion_matrix_<TEST_SECTION>.csv`: confusion matrix.
- `metrics_<TEST_SECTION>.json`: accuracy, ARI, and V-measure.

## Evaluation logic
The benchmark reduces the abundance matrix to a single predicted label per cell by taking the maximum-abundance cell type. Reported metrics are:
- accuracy;
- adjusted Rand index (ARI);
- V-measure.

This is a pragmatic evaluation strategy for consistency with label-transfer methods, but it discards uncertainty and compositional structure. For methods such as `cell2location`, that simplification should be interpreted cautiously.

## Dependencies
The pipeline imports:
- `cell2location`
- `scanpy`
- `torch`
- `numpy`, `pandas`, `scipy`
- `scikit-learn`

GPU acceleration is strongly preferable for practical runtime, although the scripts do not enforce a specific device configuration.

## Notes
- The pipeline sanitizes problematic column names before writing H5AD outputs.
- The benchmark assumes that training and test sections already share the same feature space.
- `run_cell2loc_merfish.py` appears to be a development script retained for provenance; the sweep should use `run_cell2l_sweep.sh` + `cell2l_pipeline.py`.
