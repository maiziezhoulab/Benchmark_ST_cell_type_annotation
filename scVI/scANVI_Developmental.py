import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scvi
import seaborn as sns
import torch
import os.path as osp

# ---------------- basic setup ----------------
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

base_dir = '/maiziezhou_lab2/yuling/label_Transfer/scVI/Development'
os.makedirs(base_dir, exist_ok=True)

# ---------------- load data ----------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development.h5ad')

stage_54 = data[data.obs['Batch'] == 'Stage54_telencephalon_rep2_DP8400015649BRD6_2'].copy()
stage_44 = data[data.obs['Batch'] == 'Stage44_telencephalon_rep2_FP200000239BL_E4'].copy()

stage_54.obs['Annotation'] = stage_54.obs['Annotation'].astype('category')
stage_44.obs['Annotation'] = stage_44.obs['Annotation'].astype('category')

# ---------------- annotate tech ----------------
stage_44.obs['tech'] = 'st'
stage_54.obs['tech'] = 'sc'

stage_54.layers['counts'] = stage_54.X.copy()

adata = anndata.concat([stage_44, stage_54])
adata.layers["counts"] = adata.X.copy()

# ---------------- preprocessing ----------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=3000,
    layer="counts",
    batch_key="tech",
    subset=True,
)

# Ensure counts layer is raw counts for scVI
adata.layers["counts"] = adata.X.copy()

# ---------------- timing & memory start ----------------
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

t_start = time.perf_counter()

# ===================== SCVI =====================
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="tech"
)

scvi_model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_latent=30,
    gene_likelihood="nb",
    use_batch_norm="none",
    use_layer_norm="both",
)

scvi_model.train()

adata.obsm["X_scVI"] = scvi_model.get_latent_representation()

# ===================== SCANVI =====================
SCANVI_CELLTYPE_KEY = "celltype_scanvi"
adata.obs[SCANVI_CELLTYPE_KEY] = "Unknown"

sc_mask = adata.obs["tech"] == "sc"
adata.obs.loc[sc_mask, SCANVI_CELLTYPE_KEY] = adata.obs.loc[sc_mask, "Annotation"]

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=adata,
    labels_key=SCANVI_CELLTYPE_KEY,
    unlabeled_category="Unknown",
)

scanvi_model.train(
    max_epochs=20,
    n_samples_per_label=100,
    batch_size=256
)

adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation()
adata.obs["C_scANVI"] = scanvi_model.predict()

# ---------------- timing & memory end ----------------
torch.cuda.synchronize()
t_end = time.perf_counter()

runtime_sec = t_end - t_start
peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

# ---------------- post-processing ----------------
adata.obs["C_scANVI"] = pd.Categorical(
    adata.obs["C_scANVI"].values,
    categories=adata.obs["Annotation"].cat.categories
)

pred = adata[adata.obs[SCANVI_CELLTYPE_KEY] == "Unknown"].copy()
pred.obs.to_csv(
    osp.join(base_dir, "label_transfer_1.csv"),
    index=True
)

# ---------------- save runtime & memory ----------------
runtime_df = pd.DataFrame([{
    "Elapsed_Time_sec": runtime_sec,
    "Peak_RAM_Used_MiB": peak_mem_mib
}])

runtime_df.to_csv(
    osp.join(base_dir, "runtimeSec_memoryMiB.csv"),
    index=False
)

print(runtime_df)
