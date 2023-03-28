import numpy as np
import copy
import torch
import dgl
from utils import set_seed
"""
1. Train and eval
"""


def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        logits = model.inference(data, feats)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return out, loss.item(), score

def eval_on_train_val_test_data_gnn(model, data_eval, feats, labels, criterion, evaluator, idx_train,idx_val,idx_test):
    out, loss_train, score_train = evaluate(
        model, data_eval, feats, labels, criterion, evaluator, idx_train
    )
    # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
    loss_val = criterion(out[idx_val], labels[idx_val]).item()
    score_val = evaluator(out[idx_val], labels[idx_val])
    loss_test = criterion(out[idx_test], labels[idx_test]).item()
    score_test = evaluator(out[idx_test], labels[idx_test])
    return loss_train, score_train,loss_val,score_val,loss_test,score_test

def eval_on_val_test_inductive_gnn(        
        model,
        obs_data_eval,
        obs_feats,
        obs_labels,
        data_eval,
        feats,
        labels,
        criterion,
        evaluator,
        idx_obs,
        obs_idx_val,
        obs_idx_test,
        idx_test_ind,
        logger
    ):
    obs_out, _, score_val = evaluate(
        model,
        obs_data_eval,
        obs_feats,
        obs_labels,
        criterion,
        evaluator,
        obs_idx_val,
    )
    out, _, score_test_ind = evaluate(
        model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
    )
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    return out, score_val, score_test_tran, score_test_ind
