import torch
import torch.nn as nn
# Force CPU execution to avoid CUDA library issues
#torch.cuda.is_available = lambda: False
from torch.utils.data import TensorDataset, DataLoader
import os.path as osp 
import anndata as ad
#import transfer
import sys
import os
# Add the spatialid directory to Python path
sys.path.append('/maiziezhou_lab2/yuling/label_Transfer/spatialID/SpatialID/spatialid')
# conda activate yolov8
#from transfer import Transfer
from spatialid.transfer import Transfer
import scanpy as sc
#from transfer_st_Align import Transfer
from sklearn.model_selection import train_test_split
import pandas as pd
slice_ids = ["2", "3", "4", "5", "6", "7", "9", "11", "17", "18", "19", "23", "24", "25", "26", "28", "33", "34", "36"]
def load_HMlymphNode(root_dir = '/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad', section_id =  "1"):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)  # Convert section_id to integer
    slice1 = adataT[adataT.obs['n_section'] == section_id]
    if 'gene_name' not in slice1.var.columns:
        slice1.var['gene_name'] = slice1.var_names
    slice1.obs['original_clusters'] = slice1.obs['annotation']
    slice1.obs['batch'] = section_id
    return slice1
section_ids = [4, 10]
 #----------------
sc_adata = load_HMlymphNode(section_id= slice_ids[4])

#outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/output"
#os.makedirs(outdir, exist_ok=True)
query_data = load_HMlymphNode(section_id= slice_ids[10])
query_data.write('/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/HumanLymph/spatial_data.h5ad')
set_seeds = [10, 11, 12, 13, 14]
for k in set_seeds:
    total_data = sc_adata
    single_cell_dir = osp.join('/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/HumanLymph', str(k))
    os.makedirs(single_cell_dir, exist_ok=True) 
    #total_data.write(osp.join(single_cell_dir, "single_cell_data.h5ad"))
    single_path = osp.join(single_cell_dir, 'single_cell_data.h5ad')
    total_data.write_h5ad(single_path)
    transfer_tool = Transfer(
        spatial_data = '/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/HumanLymph/spatial_data.h5ad',  
        single_data = single_path,  
        output_path = single_cell_dir, 
        device=0  
    )
    transfer_tool.learn_sc(
        filter_mt = True,        
        min_cell = 300,          
        min_gene = 10,          
        max_cell = 98.0,         
        ann_key = "original_clusters",   
        batch_size = 4096,
        epoch = 200,
        lr = 3e-4
    )
    transfer_tool.sc2st()

    transfer_tool.annotation(
        pca_dim=200,
        n_neigh=30,
        epochs=200,
        lr=0.01,
        show_results=True
    )
    print("Analysis completed! Check results in ./spatialID_results/")