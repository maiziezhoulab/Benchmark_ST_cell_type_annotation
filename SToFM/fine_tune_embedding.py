import argparse
import json
import os
import random
import sys
import math
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, "/maiziezhou_lab2/yuling/SToFM/geneformer_001")

from model.extraction import SToFM_Collator, encode_cell, load_data
from model.se2transformer import SToFMConfig, SToFMModel
from transformers import BertModel, get_linear_schedule_with_warmup

'''
Example Usage:

1. head tuning (freeze backbone, train MLP head only):
   CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 fine_tune_embedding.py \
     --task finetune \
     --train_root /maiziezhou_lab2/yuling/SToFM/datasets/train \
     --test_root /maiziezhou_lab2/yuling/SToFM/datasets/test \
     --label_key original_clusters \
     --epochs 100 --lr 1e-3 \
     --classifier_type mlp --classifier_hidden 512 --classifier_dropout 0.3 \
     --use_class_weights \
     --eval_every 5 --patience 15 \
     --save_dir /maiziezhou_lab2/yuling/SToFM/fine_tuned_ddp

2. Partial fine-tuning (unfreeze last N backbone layers + MLP head):
   CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 fine_tune_embedding.py \
     --task finetune \
     --train_root /maiziezhou_lab2/yuling/SToFM/datasets/train \
     --test_root /maiziezhou_lab2/yuling/SToFM/datasets/test \
     --label_key original_clusters \
     --epochs 100 --lr 5e-5 --classifier_lr 1e-3 \
     --unfreeze_layers 3 \
     --classifier_type mlp --classifier_hidden 512 --classifier_dropout 0.3 \
     --use_class_weights \
     --eval_every 5 --patience 15 \
     --save_dir /maiziezhou_lab2/yuling/SToFM/fine_tuned_ddp

'''

# ======================================================================
# Loss functions
# ======================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Helps focus on hard-to-classify and minority-class examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ======================================================================
# MLP Classification Head
# ======================================================================
class MLPClassifier(nn.Module):
    """
    Multi-layer classification head with dropout and LayerNorm.
    Much more expressive than a single Linear layer.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.3, num_hidden_layers: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ======================================================================
# Utilities
# ======================================================================
def build_data_info(root: str) -> Dict[str, str]:
    return {
        "data_root": root,
        "data_path": os.path.join(root, "data.h5ad"),
        "spatial_path": None,
        "model_input_path": os.path.join(root, "hf.dataset"),
        "emb_path": os.path.join(root, "ce_emb.npy"),
    }


def ensure_cell_embeddings(
    cell_encoder: BertModel,
    data_info: Dict[str, str],
    batch_size: int = 32,
    is_master: bool = True,
) -> None:
    if os.path.exists(data_info["emb_path"]):
        return
    if is_master:
        encode_cell(
            cell_encoder,
            data_info["model_input_path"],
            data_info["emb_path"],
            save=True,
            batch_size=batch_size,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_filtered_adata(data_info: Dict[str, str]) -> sc.AnnData:
    if data_info["data_path"].endswith(".h5ad"):
        adata = sc.read_h5ad(data_info["data_path"])
    else:
        adata = sc.read_10x_mtx(data_info["data_path"])
    if not os.path.exists(data_info["emb_path"]):
        raise FileNotFoundError(f"Missing embeddings at {data_info['emb_path']}")
    adata.obsm["cell_emb"] = np.load(data_info["emb_path"])
    sc.pp.filter_cells(adata, min_genes=0)
    return adata


def prepare_labels(
    adata: sc.AnnData, label_key: str, label2id: Optional[Dict[str, int]] = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    if label_key not in adata.obs.columns:
        raise ValueError(f"Column '{label_key}' not found in adata.obs")
    labels_series = adata.obs[label_key].astype(str)
    labels = labels_series.to_numpy()
    if label2id is None:
        unique_labels = sorted(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
    unknown = set(labels) - set(label2id.keys())
    if unknown:
        raise ValueError(f"Found unknown labels: {unknown}")
    label_ids = np.array([label2id[label] for label in labels], dtype=np.int64)
    return label_ids, label2id


def get_class_weights(labels: np.ndarray, num_classes: int, method: str = 'balanced') -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    unique_labels = np.unique(labels)
    if method == 'balanced':
        weights_arr = compute_class_weight('balanced', classes=unique_labels, y=labels)
    elif method == 'inverse':
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights_arr = 1.0 / counts[unique_labels]
        weights_arr = weights_arr * num_classes / weights_arr.sum()
    elif method == 'sqrt_inverse':
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights_arr = 1.0 / np.sqrt(counts[unique_labels])
        weights_arr = weights_arr * num_classes / weights_arr.sum()
    else:
        raise ValueError(f"Unknown method: {method}")

    full_weights = np.ones(num_classes, dtype=np.float32)
    for i, label in enumerate(unique_labels):
        full_weights[label] = weights_arr[i]
    return torch.tensor(full_weights, dtype=torch.float32)


def print_label_distribution(labels: np.ndarray, label2id: Dict[str, int], name: str):
    """Print label distribution for debugging."""
    id2label = {v: k for k, v in label2id.items()}
    counts = Counter(labels)
    total = len(labels)
    print(f"\n[INFO] {name} label distribution ({total} cells, {len(counts)} classes):")
    for label_id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        label_name = id2label.get(label_id, f"ID_{label_id}")
        print(f"  {label_name:40s}: {count:6d} ({count/total*100:.1f}%)")


# ======================================================================
# Embedding-only path (task=embed)
# ======================================================================
def run_embedding(
    model: SToFMModel,
    cell_encoder: BertModel,
    data_infos,
    device: torch.device,
    batch_size: int,
    split_num: int,
    leiden_res: float,
    leiden_alpha: float,
    output_filename: str,
    run_clustering: bool,
    cluster_neighbors: int,
    cluster_resolution: float,
    cluster_key: str,
    cluster_output: Optional[str],
    is_master: bool,
) -> None:
    model.eval()
    for data_info in data_infos:
        print(f"Encode cell {data_info['data_path']}")
        ensure_cell_embeddings(cell_encoder, data_info, batch_size=32, is_master=is_master)

        print(f"Load {data_info['data_path']}")
        graphs = load_data(
            **data_info,
            new_emb=False,
            device=0 if device.type == "cpu" else device.index or 0,
            filter=False,
            split_num=split_num,
            leiden_res=leiden_res,
            alpha=leiden_alpha,
        )

        if data_info["data_path"].endswith(".h5ad"):
            adata = sc.read_h5ad(data_info["data_path"])
        else:
            adata = sc.read_10x_mtx(data_info["data_path"])
        data_num = len(adata)
        print(f"Cell {data_num}, Sub-slice {len(graphs)}")

        dataloader = DataLoader(
            graphs,
            collate_fn=SToFM_Collator(mask=False, mask_pair=False),
            batch_size=batch_size,
            shuffle=False,
        )

        embeddings = torch.zeros(data_num, model.config.hidden_size)
        with torch.no_grad():
            for graph in tqdm(dataloader, desc="Get embedding"):
                indices_cpu = graph["indices"]
                graph = {k: v.to(device) for k, v in graph.items()}
                output = model(**graph)
                node_rep = output["last_hidden_state"].detach().cpu()
                mask = indices_cpu != -1
                embeddings[indices_cpu[mask]] = node_rep[mask]

        embeddings_np = embeddings.numpy()
        np.save(os.path.join(data_info["data_root"], output_filename), embeddings_np)

        if run_clustering:
            adata.obsm["stofm_emb"] = embeddings_np
            sc.pp.neighbors(adata, use_rep="stofm_emb", n_neighbors=cluster_neighbors)
            sc.tl.leiden(adata, resolution=cluster_resolution, key_added=cluster_key)
            output_path = (
                cluster_output
                if cluster_output is not None
                else os.path.join(data_info["data_root"], f"{cluster_key}.csv")
            )
            adata.obs[[cluster_key]].to_csv(output_path)
            print(f"Saved clustering assignments to {output_path}")


# ======================================================================
# Eval helper
# ======================================================================
def run_model_on_graphs(
    model: SToFMModel,
    classifier: nn.Module,
    graphs,
    num_cells: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[float]]:
    dataloader = DataLoader(
        graphs,
        collate_fn=SToFM_Collator(mask=False, mask_pair=False),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = np.full(num_cells, -1, dtype=np.int64)
    total_correct = 0
    total = 0
    model.eval()
    classifier.eval()

    with torch.no_grad():
        for batch in dataloader:
            indices_cpu = batch["indices"].clone()
            cell_labels = batch.get("cell_labels")
            inputs = {k: v.to(device) for k, v in batch.items() if k != "cell_labels"}
            outputs = model(**inputs)
            node_rep = outputs["last_hidden_state"]
            flat_rep = node_rep.reshape(-1, node_rep.shape[-1])
            flat_indices_device = inputs["indices"].reshape(-1)
            valid_mask = flat_indices_device != -1
            logits = classifier(flat_rep[valid_mask])
            preds_device = logits.argmax(dim=-1)

            flat_indices_cpu = indices_cpu.reshape(-1)
            valid_mask_cpu = flat_indices_cpu != -1
            target_indices = flat_indices_cpu[valid_mask_cpu].to(torch.long).numpy()
            preds[target_indices] = preds_device.cpu().numpy()

            if cell_labels is not None:
                labels_device = cell_labels.to(device).reshape(-1)[valid_mask]
                total_correct += (preds_device == labels_device).sum().item()
                total += labels_device.numel()

    accuracy = total_correct / total if total > 0 else None
    return preds, accuracy


def evaluate_detailed(preds, labels, label2id, name="Test"):
    """Print detailed per-class evaluation metrics."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    # Overall
    acc = np.mean(preds == labels)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    print(f"\n{'='*60}")
    print(f"[{name}] Overall accuracy:    {acc:.4f}")
    print(f"[{name}] Balanced accuracy:   {bal_acc:.4f}")
    print(f"[{name}] F1 (macro):          {f1_macro:.4f}")
    print(f"[{name}] F1 (weighted):       {f1_weighted:.4f}")

    # Per-class
    unique_preds = len(set(preds))
    print(f"[{name}] Predicted {unique_preds}/{num_labels} classes")

    correct_per_class = {}
    total_per_class = {}
    pred_per_class = Counter(preds)
    for true_id, pred_id in zip(labels, preds):
        total_per_class[true_id] = total_per_class.get(true_id, 0) + 1
        if true_id == pred_id:
            correct_per_class[true_id] = correct_per_class.get(true_id, 0) + 1

    print(f"\n  {'Class':40s} {'Acc':>8s} {'Correct/Total':>15s} {'Predicted':>10s}")
    print(f"  {'-'*40} {'-'*8} {'-'*15} {'-'*10}")
    for label, label_id in sorted(label2id.items(), key=lambda x: total_per_class.get(x[1], 0), reverse=True):
        total = total_per_class.get(label_id, 0)
        correct = correct_per_class.get(label_id, 0)
        predicted = pred_per_class.get(label_id, 0)
        class_acc = correct / total if total > 0 else 0.0
        print(f"  {label:40s} {class_acc:8.4f} {correct:>6d}/{total:<6d}   {predicted:>10d}")

    if unique_preds < num_labels * 0.5:
        print(f"\n  [WARNING] Mode collapse detected! Only {unique_preds}/{num_labels} classes predicted.")

    return {"acc": acc, "bal_acc": bal_acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# ======================================================================
# Finetune path (task=finetune)
# ======================================================================
def run_finetune(
    model: nn.Module,
    cell_encoder: BertModel,
    device: torch.device,
    args,
    is_master: bool,
    world_size: int,
    use_ddp: bool,
    local_rank: int = 0,
) -> None:
    if args.train_root is None or args.test_root is None:
        raise ValueError("Both --train_root and --test_root must be provided.")
    if args.label_key is None:
        raise ValueError("--label_key must be provided.")

    train_info = build_data_info(args.train_root)
    test_info = build_data_info(args.test_root)

    for info in (train_info, test_info):
        ensure_cell_embeddings(cell_encoder, info, batch_size=32, is_master=is_master)

    # --- Load data and labels ---
    train_adata = load_filtered_adata(train_info)
    train_labels, label2id = prepare_labels(train_adata, args.label_key)
    num_labels = len(label2id)

    test_adata = load_filtered_adata(test_info)
    if args.label_key in test_adata.obs.columns:
        test_labels, _ = prepare_labels(test_adata, args.label_key, label2id)
    else:
        test_labels = None

    if is_master:
        print_label_distribution(train_labels, label2id, "Train")
        if test_labels is not None:
            print_label_distribution(test_labels, label2id, "Test")

    # --- Build graphs ---
    train_graphs = load_data(
        **train_info,
        new_emb=False,
        device=0 if device.type == "cpu" else device.index or 0,
        filter=False,
        split_num=args.split_num,
        leiden_res=args.leiden_res,
        alpha=args.leiden_alpha,
        labels=train_labels,
    )
    test_graphs = load_data(
        **test_info,
        new_emb=False,
        device=0 if device.type == "cpu" else device.index or 0,
        filter=False,
        split_num=args.split_num,
        leiden_res=args.leiden_res,
        alpha=args.leiden_alpha,
        labels=test_labels if test_labels is not None else None,
    )

    # --- DDP sampler ---
    if use_ddp:
        train_sampler = DistributedSampler(
            train_graphs, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_graphs,
        collate_fn=SToFM_Collator(mask=False, mask_pair=False),
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    # --- Setup backbone freezing ---
    backbone = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    num_layers = len(backbone.encoder.layers)

    # First freeze everything
    for param in backbone.parameters():
        param.requires_grad = False

    # Then selectively unfreeze last N layers
    if args.unfreeze_layers > 0:
        unfreeze_n = min(args.unfreeze_layers, num_layers)
        for layer in backbone.encoder.layers[-unfreeze_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        if is_master:
            print(f"\n[CONFIG] Partial fine-tuning: unfreezing last {unfreeze_n}/{num_layers} backbone layers")
    else:
        if is_master:
            print(f"\n[CONFIG] Linear probing: entire backbone frozen ({num_layers} layers)")

    # --- Build classifier ---
    hidden_size = backbone.config.hidden_size
    if args.classifier_type == "mlp":
        classifier = MLPClassifier(
            input_dim=hidden_size,
            num_classes=num_labels,
            hidden_dim=args.classifier_hidden,
            dropout=args.classifier_dropout,
            num_hidden_layers=args.classifier_num_layers,
        ).to(device)
    else:
        classifier = nn.Linear(hidden_size, num_labels).to(device)

    if is_master:
        print(f"[CONFIG] Classifier type: {args.classifier_type}")
        if args.classifier_type == "mlp":
            print(f"  hidden_dim={args.classifier_hidden}, dropout={args.classifier_dropout}, "
                  f"num_layers={args.classifier_num_layers}")

    # --- DDP wrap classifier ---
    if use_ddp:
        classifier = nn.parallel.DistributedDataParallel(
            classifier,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # --- Optimizer with separate LRs for backbone vs classifier ---
    classifier_module = classifier.module if isinstance(classifier, nn.parallel.DistributedDataParallel) else classifier
    classifier_params = list(classifier_module.parameters())

    param_groups = []
    # Backbone trainable params (if any layers unfrozen)
    backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
    if backbone_trainable:
        param_groups.append({
            'params': backbone_trainable,
            'lr': args.lr,
            'name': 'backbone',
        })
    # Classifier params (potentially higher LR)
    cls_lr = args.classifier_lr if args.classifier_lr is not None else args.lr
    param_groups.append({
        'params': classifier_params,
        'lr': cls_lr,
        'name': 'classifier',
    })

    all_trainable = backbone_trainable + classifier_params

    if is_master:
        n_backbone_trainable = sum(p.numel() for p in backbone_trainable)
        n_classifier = sum(p.numel() for p in classifier_params)
        n_total = sum(p.numel() for p in backbone.parameters())
        print(f"\n[CONFIG] Parameter statistics:")
        print(f"  Backbone total:     {n_total:>12,}")
        print(f"  Backbone trainable: {n_backbone_trainable:>12,} ({100*n_backbone_trainable/n_total:.2f}%)")
        print(f"  Classifier:         {n_classifier:>12,}")
        print(f"  Total trainable:    {n_backbone_trainable + n_classifier:>12,}")
        print(f"  Backbone LR:        {args.lr}")
        print(f"  Classifier LR:      {cls_lr}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # --- Loss function ---
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(train_labels, num_labels, args.class_weight_method)
        class_weights = class_weights.to(device)
        if is_master:
            id2label = {v: k for k, v in label2id.items()}
            print(f"\n[CONFIG] Class weights (method={args.class_weight_method}):")
            for label_id in range(num_labels):
                label_name = id2label.get(label_id, f"ID_{label_id}")
                print(f"  {label_name:40s}: {class_weights[label_id]:.4f}")

    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        if is_master:
            print(f"[CONFIG] Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if is_master:
            wt_str = f"weighted ({args.class_weight_method})" if class_weights is not None else "unweighted"
            print(f"[CONFIG] Using CrossEntropyLoss ({wt_str})")

    # --- LR scheduler ---
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    t_total = steps_per_epoch * args.epochs
    warmup = args.warmup_steps if args.warmup_steps > 0 else int(t_total * 0.05)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=t_total)

    if is_master:
        print(f"\n[CONFIG] Scheduler: linear warmup ({warmup} steps) + linear decay ({t_total} total steps)")
        print(f"[CONFIG] Epochs={args.epochs}, Eval every={args.eval_every}, Patience={args.patience}")
        print(f"{'='*60}\n")

    # --- Training loop with early stopping ---
    best_metric = -1.0  # track best balanced accuracy on test
    patience_counter = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Set modes
        if args.unfreeze_layers > 0:
            model.train()
        else:
            model.eval()  # backbone frozen -> eval mode (BN, dropout)
        classifier.train()

        step_count = 0
        optimizer.zero_grad()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        if is_master:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        else:
            pbar = train_loader

        for batch in pbar:
            step_count += 1
            cell_labels = batch["cell_labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "cell_labels"}

            if args.unfreeze_layers > 0:
                outputs = model(**inputs)
            else:
                with torch.no_grad():
                    outputs = model(**inputs)

            node_rep = outputs["last_hidden_state"]
            # Detach if backbone is frozen (saves memory by cutting gradient graph)
            if args.unfreeze_layers == 0:
                node_rep = node_rep.detach()

            flat_rep = node_rep.reshape(-1, node_rep.shape[-1])
            flat_labels = cell_labels.reshape(-1)
            flat_indices = inputs["indices"].reshape(-1)
            valid_mask = flat_indices != -1

            logits = classifier(flat_rep[valid_mask])
            targets = flat_labels[valid_mask]

            if targets.numel() == 0:
                continue

            loss = criterion(logits, targets) / args.grad_accum_steps
            loss.backward()

            running_loss += loss.item() * args.grad_accum_steps
            preds_batch = logits.argmax(dim=-1)
            running_correct += (preds_batch == targets).sum().item()
            running_total += targets.numel()

            if step_count % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Flush remaining gradients
        if step_count % args.grad_accum_steps != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if is_master:
            epoch_loss = running_loss / max(step_count, 1)
            epoch_acc = running_correct / running_total if running_total > 0 else 0.0
            lr_backbone = optimizer.param_groups[0]['lr'] if backbone_trainable else 0.0
            lr_cls = optimizer.param_groups[-1]['lr']
            print(f"[Epoch {epoch + 1}] loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}  "
                  f"lr_backbone={lr_backbone:.2e}  lr_cls={lr_cls:.2e}")

        # --- Periodic evaluation ---
        if is_master and args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            bare_backbone = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            bare_classifier = classifier.module if isinstance(classifier, nn.parallel.DistributedDataParallel) else classifier

            test_preds, test_acc = run_model_on_graphs(
                bare_backbone, bare_classifier, test_graphs, test_adata.n_obs, args.batch_size, device
            )
            if test_labels is not None and test_acc is not None:
                from sklearn.metrics import balanced_accuracy_score, f1_score
                bal_acc = balanced_accuracy_score(test_labels, test_preds)
                f1_m = f1_score(test_labels, test_preds, average='macro', zero_division=0)
                unique_preds = len(set(test_preds))
                print(f"  [EVAL] test_acc={test_acc:.4f}  bal_acc={bal_acc:.4f}  "
                      f"f1_macro={f1_m:.4f}  classes_predicted={unique_preds}/{num_labels}")

                # Save best model by balanced accuracy
                if bal_acc > best_metric:
                    best_metric = bal_acc
                    patience_counter = 0
                    torch.save(bare_backbone.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                    torch.save(bare_classifier.state_dict(), os.path.join(args.save_dir, "best_classifier.pt"))
                    print(f"  [EVAL] New best! bal_acc={bal_acc:.4f} (saved)")
                else:
                    patience_counter += 1
                    print(f"  [EVAL] No improvement ({patience_counter}/{args.patience})")

                if patience_counter >= args.patience:
                    print(f"  [EARLY STOP] No improvement for {args.patience} eval rounds. Stopping.")
                    break

    # ====== Final evaluation & save (master only) ======
    if is_master:
        bare_backbone = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        bare_classifier = classifier.module if isinstance(classifier, nn.parallel.DistributedDataParallel) else classifier

        # Save final model
        torch.save(bare_backbone.state_dict(), os.path.join(args.save_dir, "model.pt"))
        torch.save(bare_classifier.state_dict(), os.path.join(args.save_dir, "classifier.pt"))
        with open(os.path.join(args.save_dir, "label2id.json"), "w") as f:
            json.dump(label2id, f, indent=2)

        # Load best model if it exists
        best_model_path = os.path.join(args.save_dir, "best_model.pt")
        best_cls_path = os.path.join(args.save_dir, "best_classifier.pt")
        if os.path.exists(best_model_path):
            print("\n[INFO] Loading best model for final evaluation...")
            bare_backbone.load_state_dict(torch.load(best_model_path, map_location=device))
            bare_classifier.load_state_dict(torch.load(best_cls_path, map_location=device))
        else:
            print("\n[INFO] Using final model for evaluation (no best checkpoint found)")

        # Train eval
        train_preds, train_acc = run_model_on_graphs(
            bare_backbone, bare_classifier, train_graphs, train_adata.n_obs, args.batch_size, device
        )
        if train_acc is not None:
            evaluate_detailed(train_preds, train_labels, label2id, "Train")

        # Test eval
        test_preds, test_acc = run_model_on_graphs(
            bare_backbone, bare_classifier, test_graphs, test_adata.n_obs, args.batch_size, device
        )
        if test_labels is not None and test_acc is not None:
            metrics = evaluate_detailed(test_preds, test_labels, label2id, "Test")
            with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        # Save predictions
        id2label = {v: k for k, v in label2id.items()}
        pred_labels = [id2label[int(idx)] for idx in test_preds]
        pred_column = f"pred_{args.label_key}"
        test_adata.obs[pred_column] = pred_labels

        pred_output = (
            args.pred_output
            if args.pred_output is not None
            else os.path.join(test_info["data_root"], f"{pred_column}.csv")
        )
        test_adata.obs[[pred_column]].to_csv(pred_output)
        print(f"\nSaved predictions to {pred_output}")

        np.save(os.path.join(args.save_dir, "test_predictions.npy"), test_preds)
        test_adata.write(os.path.join(args.save_dir, "test_with_predictions.h5ad"))


# ======================================================================
# Pooler for cell encoder
# ======================================================================
class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="SToFM Fine-tuning for Cell Type Prediction")

    # --- Model paths ---
    parser.add_argument("--cell_encoder_path", type=str,
                        default="/maiziezhou_lab2/yuling/SToFM/ckpt/cell_encoder")
    parser.add_argument("--config_path", type=str,
                        default="/maiziezhou_lab2/yuling/SToFM/ckpt/config.json")
    parser.add_argument("--model_path", type=str,
                        default="/maiziezhou_lab2/yuling/SToFM/ckpt/se2transformer.pth")

    # --- Data ---
    parser.add_argument("--data_path", type=str, default="/maiziezhou_lab2/yuling/SToFM/datasets")
    parser.add_argument("--output_filename", type=str, default="stofm_emb.npy")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--split_num", type=int, default=1000)
    parser.add_argument("--leiden_res", type=float, default=1.0)
    parser.add_argument("--leiden_alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # --- Clustering (embed task) ---
    parser.add_argument("--run_clustering", action="store_true")
    parser.add_argument("--cluster_resolution", type=float, default=1.0)
    parser.add_argument("--cluster_neighbors", type=int, default=15)
    parser.add_argument("--cluster_key", type=str, default="stofm_leiden")
    parser.add_argument("--cluster_output", type=str, default=None)

    # --- Task ---
    parser.add_argument("--task", choices=["embed", "finetune"], default="embed")
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument("--test_root", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=None)

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for backbone (when unfrozen)")
    parser.add_argument("--classifier_lr", type=float, default=None,
                        help="Separate LR for classifier head (default: same as --lr)")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Warmup steps (0 = auto 5%% of total)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- Backbone freezing ---
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                        help="Number of backbone layers to unfreeze from the end (0 = linear probing)")

    # --- Classifier head ---
    parser.add_argument("--classifier_type", choices=["linear", "mlp"], default="mlp",
                        help="Type of classification head")
    parser.add_argument("--classifier_hidden", type=int, default=512,
                        help="Hidden dimension for MLP classifier")
    parser.add_argument("--classifier_dropout", type=float, default=0.3,
                        help="Dropout rate for MLP classifier")
    parser.add_argument("--classifier_num_layers", type=int, default=1,
                        help="Number of hidden layers in MLP classifier")

    # --- Class imbalance ---
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class weights to handle imbalanced data")
    parser.add_argument("--class_weight_method", type=str, default="sqrt_inverse",
                        choices=["balanced", "inverse", "sqrt_inverse"],
                        help="Method for computing class weights")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of CrossEntropyLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma for Focal Loss (higher = more focus on hard examples)")

    # --- Evaluation & saving ---
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate on test set every N epochs (0 = no intermediate eval)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (in eval rounds)")
    parser.add_argument("--save_dir", type=str,
                        default="/maiziezhou_lab2/yuling/SToFM/finetune")
    parser.add_argument("--pred_output", type=str, default=None)

    args = parser.parse_args()

    # --- DDP init ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank() if use_ddp else 0
    is_master = rank == 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Seeds
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    # --- Load models ---
    config = SToFMConfig.from_pretrained(args.config_path)
    model = SToFMModel(config).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    cell_encoder = BertModel.from_pretrained(f"{args.cell_encoder_path}/cell_bert")
    cell_encoder.pooler = Pooler(
        cell_encoder.config,
        pretrained_proj=f"{args.cell_encoder_path}/cell_proj.bin",
        proj_dim=256,
    )
    cell_encoder = cell_encoder.to(device)

    if args.task == "embed":
        if args.data_path.endswith(".txt"):
            dataset_paths = [
                p.strip() for p in open(args.data_path).read().strip().split("\n") if p.strip()
            ]
        elif args.data_path.endswith(".allfiles"):
            parent = args.data_path[:-9]
            dataset_paths = sorted(os.path.join(parent, p) for p in os.listdir(parent))
        else:
            dataset_paths = [p.strip() for p in args.data_path.split(",") if p.strip()]

        data_infos = [build_data_info(p) for p in dataset_paths]
        if is_master:
            run_embedding(
                model=model, cell_encoder=cell_encoder, data_infos=data_infos,
                device=device, batch_size=args.batch_size, split_num=args.split_num,
                leiden_res=args.leiden_res, leiden_alpha=args.leiden_alpha,
                output_filename=args.output_filename, run_clustering=args.run_clustering,
                cluster_neighbors=args.cluster_neighbors,
                cluster_resolution=args.cluster_resolution,
                cluster_key=args.cluster_key, cluster_output=args.cluster_output,
                is_master=is_master,
            )

    else:  # finetune
        if use_ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )

        run_finetune(
            model=model, cell_encoder=cell_encoder, device=device, args=args,
            is_master=is_master, world_size=world_size, use_ddp=use_ddp,
            local_rank=local_rank,
        )

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
