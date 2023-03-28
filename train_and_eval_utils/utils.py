import copy

def print_debug_info(epoch,loss, loss_train,loss_val,loss_test,score_train,score_val,score_test,logger, loss_and_score):
    logger.debug(
        f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
    )
    loss_and_score += [
        [
            epoch,
            loss_train,
            loss_val,
            loss_test,
            score_train,
            score_val,
            score_test,
        ]
    ]

def early_stop_counter(count,score_val,best_score_val,epoch,model):
    if score_val >= best_score_val:
        best_epoch = epoch
        best_score_val = score_val
        state = copy.deepcopy(model.state_dict())
        count = 0
    else:
        count += 1
    return count,state

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


    
    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test
