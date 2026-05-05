import os
import os.path as osp
import time
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import seaborn as sns
import resource  # ✅ for CPU peak RSS on Linux/macOS
from scvi.model import CondSCVI, DestVI

# ---------------------------
# Optional: reproducibility
# ---------------------------
def set_all_seeds(seed: int = 2024):
    scvi.settings.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# set_all_seeds(2024)

sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

save_dir = '/maiziezhou_lab2/yuling/label_Transfer/DestVI/Development_regeneration'
os.makedirs(save_dir, exist_ok=True)

# ---------------------------
# Helper: memory utils
# ---------------------------
def get_cpu_peak_rss_mb():
    """
    Return peak RSS (MB) for current process.
    Linux: ru_maxrss is in KB
    macOS: ru_maxrss is in bytes
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024 ** 2)  # bytes -> MB
    else:
        return rss / 1024.0       # KB -> MB

def reset_gpu_peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Optional: clear cached blocks before timing (can reduce noise)
        # torch.cuda.empty_cache()

def get_gpu_peak_memory_mb():
    if not torch.cuda.is_available():
        return np.nan, np.nan
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB
    return peak_alloc, peak_reserved

# ---------------------------
# Load data
# ---------------------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development/5DPIs.h5ad')
sc_adata = data[
    data.obs['Batch'].isin([
        'Injury_5DPI_rep1_SS200000147BL_D2',
        'Injury_5DPI_rep2_SS200000147BL_D2'
    ]),
].copy()
sc_adata.obs['Annotation'] = sc_adata.obs['Annotation'].astype('category')

# Query: rep3
st_data = data[
    data.obs['Batch'] == 'Injury_5DPI_rep3_SS200000147BL_D3',
].copy()
st_data.obs['Annotation'] = st_data.obs['Annotation'].astype('category')
# st_data = data[data.obs['Batch'] == 'Injury_5DPI_rep1_SS200000147BL_D2',].copy()
# st_data.layers['counts'] = st_data.X.copy()

# sc_adata = data[data.obs['Batch'] == 'Injury_5DPI_rep2_SS200000147BL_D2',].copy()
# sc_adata.layers['counts'] = sc_adata.X.copy()
# ---------------------------
# Train once (no seed loop)
# ---------------------------
# ✅ reset GPU peak counter before timing/training
reset_gpu_peak_memory_stats()

if torch.cuda.is_available():
    torch.cuda.synchronize()
t_start = time.perf_counter()

CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="Annotation")
sc_model = CondSCVI(sc_adata, weight_obs=False)
sc_model.train()

DestVI.setup_anndata(st_data, layer="counts")
st_model = DestVI.from_rna_model(st_data, sc_model)
st_model.train(max_epochs=2500)

if torch.cuda.is_available():
    torch.cuda.synchronize()
t_end = time.perf_counter()

elapsed = t_end - t_start
print(f"Runtime (sec): {elapsed:.2f}")
cpu_peak_rss_mb = get_cpu_peak_rss_mb()
gpu_peak_alloc_mb, gpu_peak_reserved_mb = get_gpu_peak_memory_mb()

print(f"CPU Peak RSS (MB): {cpu_peak_rss_mb:.2f}")
if torch.cuda.is_available():
    print(f"GPU Peak Allocated (MB): {gpu_peak_alloc_mb:.2f}")
    print(f"GPU Peak Reserved  (MB): {gpu_peak_reserved_mb:.2f}")

# ---------------------------
# Get proportions + save
# ---------------------------
st_data.obsm["proportions"] = st_model.get_proportions()

if isinstance(st_data.obsm["proportions"], pd.DataFrame):
    df = st_data.obsm["proportions"].copy()
else:
    ct_names = list(sc_adata.obs["Annotation"].astype("category").cat.categories)
    df = pd.DataFrame(
        st_data.obsm["proportions"],
        index=st_data.obs_names,
        columns=ct_names
    )

df.to_csv(osp.join(save_dir, "proportions.csv"), index=True)

# ---------------------------
# Pred label = argmax column name per row
# ---------------------------
pred_label = df.idxmax(axis=1)
true_label = st_data.obs["Annotation"].astype(str).reindex(df.index)

valid = true_label.notna()
acc = (pred_label[valid] == true_label[valid]).mean()

print(f"Accuracy: {acc:.4f}")

pred_df = pd.DataFrame({
    "spot": df.index,
    "true_Annotation": true_label.values,
    "pred_Annotation": pred_label.values
}, index=df.index)

pred_df.to_csv(osp.join(save_dir, "pred_vs_true.csv"), index=True)

# ---------------------------
# Save summary (with runtime + peak memory)
# ---------------------------
summary_df = pd.DataFrame([{
    "Time_sec": float(elapsed),
    "Peak_Memory": float(gpu_peak_reserved_mb) if torch.cuda.is_available() else np.nan,
    "accuracy": float(acc)
}])

summary_df.to_csv(osp.join(save_dir, "runtimeSec_memoryMiB.csv"), index=False)
