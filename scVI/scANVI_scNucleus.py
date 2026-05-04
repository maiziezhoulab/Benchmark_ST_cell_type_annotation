import os
import tempfile
# conda activate DestVI
import anndata as ad 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
import time
import resource
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
 
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/obj_integrated_sc_nucleus.h5ad')

scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
 
base_dir = '/maiziezhou_lab2/yuling/label_Transfer/scVI/scNucleus'
os.makedirs(base_dir, exist_ok=True)
import os.path as osp 
adata = sc.read_h5ad('/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad')
st_data = adata[adata.obs['Section ID'] == '0503_F4_C',] 
st_data.layers['counts'] = st_data.X

st_data.obs['tech'] = 'st'
sc_adata = data
sc_adata.layers['counts'] = sc_adata.X
sc_adata.obs['tech'] = 'sc'
#########################################################
# 0) copies
st_data  = st_data.copy()
sc_adata = sc_adata.copy()
st_data.obs['final_cluster_assignment'] = 'placeholder'
# 1) align genes then concat
genes = st_data.var_names.intersection(sc_adata.var_names)
st_data  = st_data[:, genes].copy()
sc_adata = sc_adata[:, genes].copy()
print('st data', st_data.X.todense())
print('sc data--', sc_adata.obs['final_cluster_assignment'].unique())
adata = ad.concat([st_data, sc_adata])

# 2) counts layer
adata.layers["counts"] = adata.X.copy()   # ensure this is raw counts.
#adata = anndata.concat([st_data, sc_adata])
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # keep full dimension safe
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=3000,
    layer="counts",
    batch_key = "tech",
    subset=True,
)
# If X currently holds raw counts:
adata.layers["counts"] = adata.X.copy()

# If X is *normalized/logged*, you must reload raw counts into a layer (e.g., "counts_raw").
# (If you don't have raw counts, SCVI will warn and training can behave badly.)

# Setup anndata for SCVI/SCANVI
# Start runtime + peak memory measurement (model section only)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()

scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",         # this must be raw counts
    batch_key="tech"   # adjust to your meta key
)

scvi_model = scvi.model.SCVI(adata, n_layers=2, n_latent=30,
    gene_likelihood="nb",
    use_batch_norm="none",     # <-- key
    use_layer_norm="both" )
scvi_model.train()
SCVI_LATENT_KEY = "X_scVI"
adata.obsm[SCVI_LATENT_KEY] = scvi_model.get_latent_representation()
sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.3)
 
# try to obtain a better latent representation/predictions by using the labels to inform the latent space. 
# all of the query cells will have "celltype_scanvi" of value 'Unknown'
SCANVI_CELLTYPE_KEY = "celltype_scanvi"
adata.obs[SCANVI_CELLTYPE_KEY] = "Unknown"
ss2_mask = adata.obs["tech"] == "sc"
 
adata.obs.loc[ss2_mask, SCANVI_CELLTYPE_KEY] = (
    adata.obs.loc[ss2_mask, 'final_cluster_assignment'].values
)
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=adata,
    unlabeled_category="Unknown",
    labels_key=SCANVI_CELLTYPE_KEY)
scanvi_model.train(max_epochs=20, 
    n_samples_per_label=100,
    batch_size=256                          # >1 !
    )
# End runtime + peak memory measurement (model section only)
elapsed_sec = time.perf_counter() - start_time
if torch.cuda.is_available():
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    # ru_maxrss is KB on Linux; convert to MiB
    peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

runtime_df = pd.DataFrame([{
    "Time": elapsed_sec,
    "Memory": peak_memory_mib,
}])
runtime_df.to_csv(osp.join(base_dir, "runtimeSec_memoryMiB.csv"), index=False)
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTION_KEY = "C_scANVI"
adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
df = scanvi_model.predict(adata)
print(df)
adata.obs[SCANVI_PREDICTION_KEY] = df
pred = adata[adata.obs[SCANVI_CELLTYPE_KEY] =='Unknown', ]
pred.obs.to_csv(osp.join(base_dir,"label_transfer.csv"), index=True) 
print(runtime_df)
print("Saved runtime/memory to:", osp.join(base_dir, "runtimeSec_memoryMiB.csv"))