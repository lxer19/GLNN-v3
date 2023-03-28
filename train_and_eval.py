import numpy as np
import copy
import torch
import dgl
from utils import set_seed
from train_and_eval_utils.mlp import train_mini_batch,evaluate_mini_batch,eval_on_train_val_test_data_mlp
from train_and_eval_utils.gnn import train,evaluate,eval_on_train_val_test_data_gnn,eval_on_val_test_inductive_gnn
from train_and_eval_utils.sage import train_sage,get_train_eval_datasets_sage
from train_and_eval_utils.utils import print_debug_info, early_stop_counter

"""
2. Run teacher
"""


def run_transductive_sage(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    idx_train, idx_val, idx_test = indices
    batch_size = conf["batch_size"]
    data,data_eval= get_train_eval_datasets_sage(g,idx_train,batch_size,conf["fan_out"],conf["num_workers"])
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_sage(model, data, feats, labels, criterion, optimizer)
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_gnn(model, data_eval, feats, labels, criterion, evaluator, idx_train,idx_val,idx_test)
            print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score)
            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    out, _, score_val = evaluate(
        model, data_eval, feats, labels, criterion, evaluator, idx_val
    )
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def run_transductive_mlp(
    conf,
    model,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    batch_size = conf["batch_size"]
    idx_train, idx_val, idx_test = indices
    feats_train, labels_train = feats[idx_train], labels[idx_train]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_mini_batch(
            model, feats_train, labels_train, batch_size, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_mlp(model, feats_train,labels_train,feats_val, labels_val,feats_test, labels_test,criterion,batch_size,evaluator)
            print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score)
            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    out, _, score_val = evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_val
    )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test

def run_transductive_gnn(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    device = conf["device"]
    idx_train, idx_val, idx_test = indices
    batch_size = conf["batch_size"]
    g = g.to(device)
    data = g
    data_eval = g
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train(model, data, feats, labels, criterion, optimizer, idx_train)
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test,
                score_test
            )=eval_on_train_val_test_data_gnn(model, data_eval, feats, labels, criterion, evaluator, idx_train,idx_val,idx_test)
            print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score)
            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    out, _, score_val = evaluate(
        model, data_eval, feats, labels, criterion, evaluator, idx_val
    )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test



def run_transductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        return run_transductive_sage(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
    elif "MLP" in model.model_name:
        return run_transductive_mlp(
            conf,
            model,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
    else:
        return run_transductive_gnn(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )

def print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,
            score_test_ind,logger, loss_and_score):
    logger.debug(
        f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
    )
    loss_and_score += [
        [
            epoch,
            loss_train,
            loss_val,
            loss_test_tran,
            loss_test_ind,
            score_train,
            score_val,
            score_test_tran,
            score_test_ind,
        ]
    ]



def run_inductive_sage(    
    conf,
    model,
    obs_g,
    g,
    obs_feats, 
    obs_labels,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score
):
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    batch_size = conf["batch_size"]
    obs_data,obs_data_eval= get_train_eval_datasets_sage(obs_g,obs_idx_train,batch_size,conf["fan_out"],conf["num_workers"])
    data,data_eval=get_train_eval_datasets_sage(g,obs_idx_train,batch_size,conf["fan_out"],conf["num_workers"])

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_sage(
            model, obs_data, obs_feats, obs_labels, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_gnn(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_train,obs_idx_val,obs_idx_test)                            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(
                model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
            )
            print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,
            score_test_ind,logger, loss_and_score)

            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    out, score_val, score_test_tran, score_test_ind = eval_on_val_test_inductive_gnn(        
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
    )
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind

def run_inductive_mlp(    
    conf,
    model,
    obs_g,
    g,
    obs_feats, 
    obs_labels,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score
):
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    batch_size = conf["batch_size"]
    feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train_mini_batch(
            model, feats_train, labels_train, batch_size, criterion, optimizer
        )
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_mlp(model, feats_train,labels_train,feats_val, labels_val,feats_test_tran, labels_test_tran,criterion,batch_size,evaluator)
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion,
                batch_size,
                evaluator,
            )
            print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,score_test_ind,logger, loss_and_score)
            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
    )
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind

def run_inductive_gnn(    
    conf,
    model,
    obs_g,
    g,
    obs_feats, 
    obs_labels,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score
):
    device = conf["device"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    batch_size = conf["batch_size"]

    obs_g = obs_g.to(device)
    g = g.to(device)

    obs_data = obs_g
    obs_data_eval = obs_g
    data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss = train(model, obs_data, obs_feats, obs_labels, criterion, optimizer, obs_idx_train)
        if epoch % conf["eval_interval"] == 0:
            (
                loss_train, 
                score_train,
                loss_val,
                score_val,
                loss_test_tran,
                score_test_tran
            )=eval_on_train_val_test_data_gnn(model, obs_data_eval, obs_feats, obs_labels, criterion, evaluator, obs_idx_train,obs_idx_val,obs_idx_test)                            # Evaluate the inductive part with the full graph
            out, loss_test_ind, score_test_ind = evaluate(
                model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
            )
            print_debug_info_inductive(epoch,loss, loss_train,loss_val,loss_test_tran,loss_test_ind,score_train,score_val,score_test_tran,
            score_test_ind,logger, loss_and_score)

            count,state=early_stop_counter(count,score_val,best_score_val,epoch,model)

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break
    model.load_state_dict(state)
    return  eval_on_val_test_inductive_gnn(        
        model,
        obs_data_eval,
        obs_feats,
        obs_labels,
        data_eval,
        feats,
        labels,
        criterion,
        evaluator,
        obs_idx_val,
        obs_idx_test,
        idx_test_ind,
        logger
    )

def run_inductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        return run_inductive_sage(    
            conf,
            model,
            obs_g,
            g,
            obs_feats, 
            obs_labels,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score
        )

    elif "MLP" in model.model_name:
        return run_inductive_mlp(    
            conf,
            model,
            obs_g,
            g,
            obs_feats, 
            obs_labels,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score
        )

    else:
        return run_inductive_gnn(    
            conf,
            model,
            obs_g,
            g,
            obs_feats, 
            obs_labels,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score
        )


"""
3. Distill
"""


def distill_run_transductive(
    conf,
    model,
    g,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    g = g.to(device)
    data = g
    data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train(
            model, data, feats, labels, criterion_l, optimizer, idx_l,lamb
        )
        loss_t = train(
            model, data, feats, out_t_all, criterion_t, optimizer, idx_t, 1 - lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            out, loss_l, score_l = evaluate(
                model, data_eval, feats, labels, criterion_l, evaluator, idx_l
            )
            # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
            loss_val = criterion_l(out[idx_val], labels[idx_val]).item()
            score_val = evaluator(out[idx_val], labels[idx_val])
            loss_test = criterion_l(out[idx_test], labels[idx_test]).item()
            score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

        if score_val >= best_score_val:
            best_epoch = epoch
            best_score_val = score_val
            state = copy.deepcopy(model.state_dict())
            count = 0
        else:
            count += 1

    model.load_state_dict(state)
    out, _, score_val = evaluate(
        model, data_eval,feats, labels, criterion_l, evaluator, idx_val
    )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels[idx_test])

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive_gnn(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
    

def distill_run_inductive_mle(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, 1 - lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind

def distill_run_inductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    if model.model_name=="MLE":
        return distill_run_inductive_mle(
            conf,
            model,
            feats,
            labels,
            out_t_all,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
