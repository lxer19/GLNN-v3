import numpy as np
import copy
import torch
import dgl
from utils import set_seed
"""
1. Train and eval
"""

def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)




def get_train_eval_datasets_sage(g,idx_train,batch_size,fan_out,num_workers):
    # Create dataloader for SAGE

    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [eval(fanout) for fanout in fan_out.split(",")]
    )
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        idx_train,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader_eval = dgl.dataloading.NodeDataLoader(
        g,
        torch.arange(g.num_nodes()),
        sampler_eval,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    data = dataloader
    data_eval = dataloader_eval
    return data,data_eval
