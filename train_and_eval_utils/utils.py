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
