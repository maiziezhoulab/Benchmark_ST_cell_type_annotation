# SPOTlight

## Overview
This folder contains a benchmark-specific `SPOTlight` workflow for leave-one-section-out cell type annotation on the MERFISH spinal cord dataset. The setup is incremental: for a chosen held-out test section, the training reference grows from 1 section to all 17 remaining sections.

The implementation follows a supervised deconvolution pattern:
1. split cells by training versus held-out section;
2. compute marker genes on the training subset;
3. train the SPOTlight NMF model on the training cells;
4. deconvolve the held-out cells;
5. convert deconvolution weights to hard labels by argmax and score against ground truth.

## Files
- `train_spolight_reference.sh`: wrapper that performs the ordered reference-size sweep for one held-out section.
- `spotlight_train_apply_score.R`: main end-to-end script for preprocessing, marker selection, model training, deconvolution, and evaluation.
- `eval.ipynb`: exploratory analysis notebook.
- `wrong_code/`: older or discarded code paths retained for reference only.

## Inputs
The workflow expects a MERFISH `.h5ad` file with at least:
- an observation column for section identity;
- an observation column for cell-type labels;
- a count matrix available as `counts` or `X`.

Default settings in `train_spolight_reference.sh`:
```bash
H5AD_PATH=/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad
LABEL_COL="MERFISH cell type annotation"
SECTION_COL="Section ID"
```

## Usage
Run the full sweep for one held-out section:
```bash
bash train_spolight_reference.sh 0503_M4_S
```

For each reference size `i`, the script trains on the first `i` sections from the ordered training pool and writes results to:
```text
spotlight_results_<TEST_SECTION>_<i>
```

## Main processing choices
Key defaults in `spotlight_train_apply_score.R`:
- top HVGs: `3000`
- markers per cluster: `100`
- maximum training cells per label after downsampling: `100`

The script performs:
- `logNormCounts` normalization on the training subset;
- HVG selection with `scran::modelGeneVar`;
- marker ranking with `scran::scoreMarkers`;
- optional class-balanced downsampling before NMF training.

These choices are operationally sensible, but they introduce a benchmark-specific bias: downsampling improves tractability and class balance, yet may suppress rare-cell heterogeneity. Any comparison to methods that use the full reference should keep that asymmetry in view.

## Outputs
Each run directory typically contains:
- `spotlight_model_train.rds`: fitted SPOTlight model.
- `topic_profiles_train.csv`: learned topic profiles.
- `proportions_test_slice.csv`: deconvolution proportions for the held-out section.
- `metrics_test_slice.csv`: summary metrics on the held-out section.

Some runs also depend on auxiliary marker files written under `spotlight_out/`.

## Evaluation logic
Predicted labels are derived by taking the maximum-weight cell type from the deconvolution matrix. The script reports:
- accuracy;
- ARI;
- V-measure.

As with other deconvolution-based methods, this hard-label reduction is convenient but lossy. It is appropriate for a classification benchmark, not for preserving the full interpretability of mixture weights.

## Dependencies
The R script imports:
- `optparse`
- `zellkonverter`
- `SingleCellExperiment`
- `SPOTlight`
- `scran`
- `scuttle`
- `Matrix`
- `dplyr`
- `mclust`
- `aricode`

## Notes
- The script resolves observation-column names robustly, including minor formatting differences.
- Training is explicitly restricted to genes shared with the held-out section.
- `wrong_code/` should not be treated as the current benchmark implementation.
