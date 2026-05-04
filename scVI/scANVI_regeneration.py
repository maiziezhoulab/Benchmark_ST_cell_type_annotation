import os
import os.path as osp
import time
import random
import resource  
# conda activate DestVI
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch

# ---------------------------
# Helpers: memory + timing
# ---------------------------
def get_cpu_peak_rss_mb():
    """
    Peak RSS for current process.
    Linux: ru_maxrss in KB
    macOS: ru_maxrss in bytes
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 ** 2)   # bytes -> MB
    return rss / 1024.0            # KB -> MB

def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # optional but good for fair benchmarking
        torch.cuda.reset_peak_memory_stats()

def get_gpu_peak_mb():
    if not torch.cuda.is_available():
        return np.nan, np.nan
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return peak_alloc, peak_reserved

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ---------------------------
# Load data
# ---------------------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development/5DPIs.h5ad')
# stage_54: rep1 + rep2
stage_54 = data[
    data.obs['Batch'].isin([
        'Injury_5DPI_rep1_SS200000147BL_D2',
        'Injury_5DPI_rep2_SS200000147BL_D2'
    ]),
].copy()
stage_54.obs['Annotation'] = stage_54.obs['Annotation'].astype('category')

# stage_44: rep3
stage_44 = data[
    data.obs['Batch'] == 'Injury_5DPI_rep3_SS200000147BL_D3',
].copy()
stage_44.obs['Annotation'] = stage_44.obs['Annotation'].astype('category')
# stage_54 = data[data.obs['Batch'] == 'Injury_5DPI_rep1_SS200000147BL_D2',].copy()
# stage_54.obs['Annotation'] = stage_54.obs['Annotation'].astype('category')
print(stage_54.X.toarray())
# stage_44 = data[data.obs['Batch'] == 'Injury_5DPI_rep2_SS200000147BL_D2',].copy()
# stage_44.obs['Annotation'] = stage_44.obs['Annotation'].astype('category')

scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

base_dir = '/maiziezhou_lab2/yuling/label_Transfer/scVI/Development_regeneration'
os.makedirs(base_dir, exist_ok=True)

# ---------------------------
# Build adata / preprocessing (NOT benchmarked)
# ---------------------------
st_data = stage_44.copy()
st_data.obs['tech'] = 'st'

sc_adata = stage_54.copy()
sc_adata.layers['counts'] = sc_adata.X.copy()
sc_adata.obs['tech'] = 'sc'

adata = anndata.concat([st_data, sc_adata])
adata.layers["counts"] = adata.X.copy()  # raw counts before normalization

# preprocessing (excluded from timing/memory benchmark)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    flavor = "seurat_v3",
    n_top_genes = 3000,
    layer="counts",      # uses raw counts layer for HVG
    batch_key="tech",
    subset=True,
)

# IMPORTANT:
# After HVG subset, ensure "counts" still points to raw-count-like matrix for the subset genes.
# If adata.X is log-normalized now, don't overwrite counts with adata.X.
# We reconstruct counts from raw if available.
if adata.raw is not None:
    # align raw to current HVG genes
    raw_df = adata.raw[:, adata.var_names].to_adata().X
    adata.layers["counts"] = raw_df.copy()
else:
    # fallback (not ideal if X is normalized/logged)
    adata.layers["counts"] = adata.X.copy()

# ---------------------------
# Model benchmark starts HERE (SCVI + SCANVI only)
# ---------------------------
reset_gpu_peak()
cuda_sync()
model_t0 = time.perf_counter()

# Setup anndata for SCVI/SCANVI (counts should be raw counts)
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="tech"
)
# ---- SCVI ----
cuda_sync()
scvi_t0 = time.perf_counter()

scvi_model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_latent=30,
    gene_likelihood="nb",
    use_batch_norm="none",
    use_layer_norm="both",
)
scvi_model.train()

cuda_sync()
scvi_t1 = time.perf_counter()
scvi_runtime_sec = scvi_t1 - scvi_t0

# downstream latent/UMAP (usually not counted as "model training")
SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = scvi_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.3)

# ---- SCANVI label prep ----
SCANVI_CELLTYPE_KEY = "celltype_scanvi"
adata.obs[SCANVI_CELLTYPE_KEY] = "Unknown"
ss2_mask = adata.obs["tech"] == "sc"
adata.obs.loc[ss2_mask, SCANVI_CELLTYPE_KEY] = adata.obs.loc[ss2_mask, "Annotation"].astype(str).values

# ---- SCANVI ----
cuda_sync()
scanvi_t0 = time.perf_counter()

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=adata,
    unlabeled_category="Unknown",
    labels_key=SCANVI_CELLTYPE_KEY,
)

scanvi_model.train(
    max_epochs=20,
    n_samples_per_label=100,
    batch_size=256,
)

cuda_sync()
scanvi_t1 = time.perf_counter()
scanvi_runtime_sec = scanvi_t1 - scanvi_t0

cuda_sync()
model_t1 = time.perf_counter()
total_model_runtime_sec = model_t1 - model_t0

# Collect peak memory AFTER all model training
cpu_peak_rss_mb = get_cpu_peak_rss_mb()
gpu_peak_alloc_mb, gpu_peak_reserved_mb = get_gpu_peak_mb()

print(f"SCVI runtime (sec):   {scvi_runtime_sec:.2f}")
print(f"SCANVI runtime (sec): {scanvi_runtime_sec:.2f}")
print(f"Total model runtime (sec): {total_model_runtime_sec:.2f}")
print(f"CPU peak RSS (MB):    {cpu_peak_rss_mb:.2f}")
if torch.cuda.is_available():
    print(f"GPU peak allocated (MB): {gpu_peak_alloc_mb:.2f}")
    print(f"GPU peak reserved  (MB): {gpu_peak_reserved_mb:.2f}")

# ---------------------------
# Predictions (not part of training benchmark)
# ---------------------------
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTION_KEY = "C_scANVI"

adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)

sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.3)

# Make categories line up
if pd.api.types.is_categorical_dtype(adata.obs["Annotation"]):
    anno_cats = list(adata.obs["Annotation"].cat.categories)
else:
    anno_cats = sorted(pd.unique(adata.obs["Annotation"].astype(str)))

adata.obs[SCANVI_PREDICTION_KEY] = pd.Categorical(
    adata.obs[SCANVI_PREDICTION_KEY].astype(str).values,
    categories=anno_cats
)
# Query (unknown = former ST)
pred = adata[adata.obs[SCANVI_CELLTYPE_KEY] == 'Unknown', :].copy()
pred.obs.to_csv(osp.join(base_dir, "label_transfer.csv"), index=True)

# Optional accuracy (if ST has Annotation)
if "Annotation" in pred.obs.columns:
    true_label = pred.obs["Annotation"].astype(str)
    pred_label = pred.obs[SCANVI_PREDICTION_KEY].astype(str)
    valid = true_label.notna()
    acc = (pred_label[valid] == true_label[valid]).mean()
    print(f"Prediction accuracy on Unknown/ST subset: {acc:.4f}")
else:
    acc = np.nan

# ---------------------------
# Save benchmark summary
# ---------------------------
summary_df = pd.DataFrame([{
    "Time_sec": float(total_model_runtime_sec),
    "Peak_Memory": float(gpu_peak_reserved_mb) if torch.cuda.is_available() else np.nan,
    "accuracy_query": float(acc) if not pd.isna(acc) else np.nan,
}])

summary_path = osp.join(base_dir, "runtimeSec_memoryMiB.csv")
summary_df.to_csv(summary_path, index=False)