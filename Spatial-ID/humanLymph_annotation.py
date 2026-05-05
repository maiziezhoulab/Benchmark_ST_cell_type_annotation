#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn.functional as F
import os
import time
import random
import argparse
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch_geometric
import os.path as osp 
#from cell_type_annotation_model import DNNModel, SpatialModelTrainer
import torch.nn as nn
from spatial_train import SpatialModelTrainer
import numpy as np
import pandas as pd
import re
import torch
#-------
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate=0.5):
        super(DNNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 2)),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(int(hidden_dim / 2), output_dim),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
        
def clean_spatial_obsm(adata, key='spatial', to_um=False):
    """Ensure adata.obsm[key] is a numeric (N,2) float32 array."""
    sp = adata.obsm[key]
    if isinstance(sp, pd.DataFrame):
        sp = sp.values
    sp = np.asarray(sp, dtype=object)
    if sp.ndim == 1:
        pts = []
        for p in sp:
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
            elif isinstance(p, str):  
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p)
                if len(nums) >= 2:
                    pts.append([float(nums[0]), float(nums[1])])
                else:
                    pts.append([np.nan, np.nan])
            else:
                pts.append([np.nan, np.nan])
        sp = np.array(pts, dtype=float)
    elif sp.ndim == 2:
        try:
            sp = sp.astype(float)
        except Exception:
            sp = np.array([[float(x) for x in row[:2]] if len(row) >= 2 else [np.nan, np.nan]
                           for row in sp], dtype=float)
    else:
        raise ValueError(f"Unexpected shape for obsm['{key}']: {sp.shape}")
    if sp.shape[1] > 2:
        sp = sp[:, :2]
    mask = ~np.any(np.isnan(sp), axis=1)
    if not np.all(mask):
        adata._inplace_subset_obs(mask)
        sp = sp[mask]
    if to_um:
        sp = sp * 1000.0
    adata.obsm[key] = sp.astype(np.float32)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


config = {
    'data': {
        'data_dir': '/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/spinal/oneSlice/',
        'save_dir': '/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/spinal/oneSlice/',
        'dataset': 'HumanLymph',
    },
    'preprocess': {
        'filter_mt': True,
        'cell_min_counts': 0,
        'gene_min_cells': 0,
        'cell_max_counts_percent': 100.0,
        'drop_rate': 0,
    },
    'transfer': {
        'dnn_model': '/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/spinal/oneSlice/learn_sc_dnn.bgi',
        'gpu': '0',
        'batch_size': 512,
    },
    'train': {
        'pca_dim': 200,  # for Stereoseq only
        'k_graph': 30,
        'edge_weight': True,
        'feat_dim': 64,
        'w_dae': 1.0,
        'w_gae': 1.0,
        'w_cls': 10.0,
        'epochs': 200,
    }
}


def spatial_classification_tool(config, data_name, save_dir, data_dir, model_path):
    ''' Spatial classification workflow.

    # Arguments
        config (Config): Configuration parameters.
        data_name (str): Data name.
    '''
    ######################################
    #         Part 1: Load data          #
    ######################################
    
    # Set path and load data.
    print('\n==> Loading data...')
    dataset = config['data']['dataset'] 
    #data_dir, save_dir = config['data']['data_dir'], config['data']['save_dir'] 
    #data_dir = config['data']['data_dir']
    print(f'  Data name: {data_name} ({dataset})')
    print(f'  Data path: {data_dir}')
    print(f'  Save path: {save_dir}')
    adata = sc.read_h5ad(os.path.join(data_dir, f'{data_name}.h5ad'))

    # Initalize save path.
    model_name = f'spatialID-{data_name}'
    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ######################################
    #         Part 2: Preprocess         #
    ######################################
    if dataset == 'HumanLymph':
        params = config['preprocess']
        if params['filter_mt']:
            adata.var['mt'] = adata.var_names.str.startswith(('Mt-', 'mt-'))
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
            adata = adata[adata.obs['pct_counts_mt'] < 10].copy()
        if params['cell_min_counts'] > 0:
            sc.pp.filter_cells(adata, min_counts=params['cell_min_counts'])
        if params['gene_min_cells'] > 0:
            sc.pp.filter_genes(adata, min_cells=params['gene_min_cells'])
        if params['cell_max_counts_percent'] < 100:
            max_counts = np.percentile(adata.X.sum(1), params['cell_max_counts_percent'])
            sc.pp.filter_cells(adata, max_counts=max_counts)
    print('\n==> Preprocessing...')
    strings = [f'{k}={v}' for k, v in config['preprocess'].items()]
    print('  Parameters(%s)' % (', '.join(strings)))
    if type(adata.X) != np.ndarray:
        adata_X_sparse_backup = adata.X.copy()
        adata.X = adata.X.toarray()
    print('  %s: %d cells × %d genes.' % (data_name, adata.shape[0], adata.shape[1]))


    ######################################
    #  Part 3: Transfer from sc-dataset  #
    ######################################

    print('\n==> Transfering from sc-dataset...')
    strings = [f'{k}={v}' for k, v in config['transfer'].items()]
    print('  Parameters(%s)' % (', '.join(strings)))

    # Set device.
    os.environ['CUDA_VISIBLE_DEVICES'] = config['transfer']['gpu']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DNN model trained by sc-dataset.
    #checkpoint = torch.load(config['transfer']['dnn_model'])
    checkpoint = torch.load(model_path)
    dnn_model = checkpoint['model'].to(device)
    print('dnn model', dnn_model)
    dnn_model.eval()

    # Initialize DNN input.
    marker_genes = checkpoint['marker_genes']
    gene_indices = adata.var_names.get_indexer(marker_genes)
    adata_X = np.pad(adata.X, ((0,0),(0,1)))[:, gene_indices]
    norm_factor = np.linalg.norm(adata_X, axis=1, keepdims=True)
    norm_factor[norm_factor == 0] = 1
    dnn_inputs = torch.Tensor(adata_X / norm_factor).split(config['transfer']['batch_size'])

    # Inference with DNN model.
    dnn_predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dnn_inputs):
            inputs = inputs.to(device)
            outputs = dnn_model(inputs)
            dnn_predictions.append(outputs.detach().cpu().numpy())
    label_names = checkpoint['label_names']
    adata.obsm['pseudo_label'] = np.concatenate(dnn_predictions)
    adata.obs['pseudo_class'] = pd.Categorical([label_names[i] for i in adata.obsm['pseudo_label'].argmax(1)])
    adata.uns['pseudo_classes'] = label_names
    print(adata.obs['pseudo_class'])
    # Compute accuracy (only for Slide-seq).
    if dataset == 'Hyp_3D':
        indices = np.where(~adata.obs['original_clusters'].isin(['Ambiguous']))[0]
        adjusted_pr = adata.obs['pseudo_class'][indices].to_numpy()
        adjusted_gt = adata.obs['original_clusters'].to_numpy()
        '''
        adjusted_gt = adata.obs['Cell_class'][indices].replace(
            ['Endothelial 1', 'Endothelial 2', 'Endothelial 3',
             'OD Immature 1', 'OD Immature 2',
             'OD Mature 1', 'OD Mature 2', 'OD Mature 3', 'OD Mature 4',
             'Astrocyte', 'Pericytes'], 
            ['Endothelial', 'Endothelial', 'Endothelial',
             'Immature oligodendrocyte', 'Immature oligodendrocyte',
             'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte',
             'Astrocytes', 'Mural']).to_numpy()
        '''
        acc = (adjusted_pr == adjusted_gt).sum() / len(indices) * 100.0
        print('  %s Acc (transfer only): %.2f%%' % (data_name, acc))


    ######################################
    #      Part 4: Train GDAE model      #
    ######################################

    print('\n==> Model training...')
    strings = [f'{k}={v}' for k, v in config['train'].items()]
    print('  Parameters(%s)' % (', '.join(strings)))

    # Normalize gene expression.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_X = (adata.X - adata.X.mean(0)) / (adata.X.std(0) + 1e-10)

    # Construct spatial graph.
    gene_mat = torch.Tensor(adata_X)
    if dataset == 'Stereoseq':  # PCA
        u, s, v = torch.pca_lowrank(gene_mat, config['train']['pca_dim'])
        gene_mat = torch.matmul(gene_mat, v)
    '''
    if dataset == 'Hyp_3D':
        z_axis = adata.obs['n_section'].to_numpy() * 1000
        for z in set(z_axis):
            adata.obsm['spatial'][z_axis == z] -= adata.obsm['spatial'][z_axis == z].min(0)
        adata.obsm['spatial'] = np.hstack([adata.obsm['spatial'], z_axis.reshape(-1, 1)])
    #cell_coo = torch.Tensor(adata.obsm['spatial'])
    '''
    clean_spatial_obsm(adata, key='spatial', to_um=False)  
    cell_coo = torch.from_numpy(adata.obsm['spatial'])  # dtype=float32，OK
    data = torch_geometric.data.Data(x=gene_mat, pos=cell_coo)
    data = torch_geometric.transforms.KNNGraph(k=config['train']['k_graph'], loop=True)(data)
    data.y = torch.Tensor(adata.obsm['pseudo_label'])

    # Make distances as edge weights.
    if config['train']['edge_weight']:
        data = torch_geometric.transforms.Distance()(data)
        data.edge_weight = 1 - data.edge_attr[:,0]
    else:
        data.edge_weight = torch.ones(data.edge_index.size(1))

    # Train self-supervision model.
    input_dim = data.num_features

# ---- from your AnnData ----
    pseudo = adata.obsm['pseudo_label']              # shape (N,C) or (N,)
    arr = np.asarray(pseudo)

# If logits/probabilities -> argmax to class ids:
    if arr.ndim == 2:
        arr = arr.argmax(axis=1)

# Convert to torch tensor
    y = torch.as_tensor(arr)

# Ensure integer dtype
    if y.dtype != torch.long:
        y = y.to(torch.long)

# If labels might start at 1 (or any non 0-based set), remap to 0..C-1:
    uni = torch.unique(y)
    if uni.min().item() != 0 or not torch.equal(uni, torch.arange(len(uni), device=uni.device)):
    # map sorted unique labels to 0..K-1
        sorted_uni, _ = torch.sort(uni)
    # vectorized remap using searchsorted
        y = torch.searchsorted(sorted_uni, y)

    data.y = y  # now dtype=long and 0-based
    num_classes = int(data.y.max().item() + 1)

    #num_classes = len(adata.uns['pseudo_classes'])
    #----------------- time and memory 
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_model_start = time.perf_counter()
    trainer = SpatialModelTrainer(input_dim, num_classes, device, config['train'])
    trainer.train(data, config['train'])
    trainer.save_checkpoint(os.path.join(save_dir, f'{model_name}.t7'))
    print('trainer------', trainer)
    # Inference.
    print('\n==> Inferencing...')
   
    #Z_query = trainer.extract_embeddings(data, which="z_cat_raw")
    # save embedding
    #np.save(save_dir + "/Z_train.npy", Z_query.cpu().numpy())
    logits = trainer.predict(data)
    predictions = logits.argmax(dim=1).cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_model_end = time.perf_counter()
    model_runtime = t_model_end - t_model_start

    gpu_alloc = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    gpu_reserved = (
        torch.cuda.max_memory_reserved() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    #celltype_pred = pd.Categorical([adata.uns['pseudo_classes'][i] for i in predictions.argmax(1)])
#########   
    pseudo = adata.uns['pseudo_classes']  
    K = len(pseudo)

    pred = predictions
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred)
    if pred.ndim > 2:
        pred = np.squeeze(pred)

    if pred.ndim == 2:
        if pred.shape[1] == K:
            idx = pred.argmax(axis=1)
        elif pred.shape[0] == K:
            idx = pred.argmax(axis=0)
        else:
            raise ValueError(f"Logits shape {pred.shape} does not match #classes={K}.")
    elif pred.ndim == 1:
        if np.issubdtype(pred.dtype, np.integer):
            idx = pred.astype(int)
        else:
            idx = (pred >= 0.5).astype(int)
    else:
        raise ValueError(f"Unexpected predictions ndim={pred.ndim}, shape={pred.shape}")
    celltype_pred = pd.Categorical([pseudo[int(i)] for i in idx])
#################

    if dataset == 'HumanLymph':
        indices = np.where(~adata.obs['original_clusters'].isin(['Ambiguous']))[0]
        adjusted_pr = celltype_pred[indices].to_numpy()
        adjusted_gt = adata.obs['original_clusters'][indices].to_numpy()
        '''
        adjusted_gt = adata.obs['Cell_class'][indices].replace(
            ['Endothelial 1', 'Endothelial 2', 'Endothelial 3',
             'OD Immature 1', 'OD Immature 2',
             'OD Mature 1', 'OD Mature 2', 'OD Mature 3', 'OD Mature 4',
             'Astrocyte', 'Pericytes'], 
            ['Endothelial', 'Endothelial', 'Endothelial',
             'Immature oligodendrocyte', 'Immature oligodendrocyte',
             'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte', 'Mature oligodendrocyte',
             'Astrocytes', 'Mural']).to_numpy()
        '''
        
        acc = (adjusted_pr == adjusted_gt).sum() / len(indices) * 100.0
        from sklearn.metrics import adjusted_rand_score as ari_score
        ari = ari_score(adjusted_pr, adjusted_gt)
        print('  %s Ari (transfer+GDAE): %.2f%%' % (data_name, ari))
        print('  %s Acc (transfer+GDAE): %.2f%%' % (data_name, acc))
    # Save results.
    result = pd.DataFrame({'cell': adata.obs_names.tolist(), 'celltype_pred': celltype_pred})
    result.to_csv(os.path.join(save_dir, f'{model_name}.csv'), index=False)
    adata.obsm['celltype_prob'] = predictions
    adata.obs['celltype_pred'] = pd.Categorical(celltype_pred)
    return acc, ari, model_runtime, gpu_alloc, gpu_reserved


if __name__ == '__main__':
    import argparse
    import os.path as osp
    import pandas as pd
    import time
    import torch

    data_list = ['sample1']
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', choices=data_list)
    args = parser.parse_args()

    path_d = '/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/HumanLymph'

    rows = []
    set_seeds = [2024]

    for i in set_seeds:
        path_s = osp.join(path_d, str(i))
        model_p = osp.join(path_s, 'learn_sc_dnn.bgi')

        acc, ari, model_runtime, gpu_alloc, gpu_reserved = spatial_classification_tool(
            config,
            data_name='spatial_data',
            save_dir=path_s,
            data_dir=path_d,
            model_path=model_p
        )
 
        rows.append({
            "Elapsed_Time_sec": model_runtime,
            "Peak_RAM_Used_MiB": gpu_reserved
        })

        
        summary = pd.DataFrame(rows)
            
        out_csv = osp.join(path_s, "runtimeSec_memoryMiB.csv")
        summary.to_csv(out_csv)

        print("\n=== SpatialID Summary ===")
        print(summary)
        print(f"\nSaved to: {out_csv}")
