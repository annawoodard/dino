"""Copy-paste with minor modifications for readability from deepCENT
https://github.com/yicjia/DeepCENT/blob/main/DeepCENT/deepcent_regular.py

"""
import numpy as np
import torch
import torch.nn as nn


def one_pair(x0, x1):
    return 1 + nn.LogSigmoid()(x1 - x0) / np.log(2.0)


def lower_bound_rank_loss(pred, obs, event):
    N = pred.size(0)
    # N x N
    all_pairs = one_pair(pred.view(N, 1), pred.view(1, N))

    well_ordered_pair = (obs.view(1, N) - obs.view(N, 1)) > 0

    # True for pairs where Y_i < Y_j and Y_i is an event; False otherwise
    lower_outcome_is_event = (event.view(1, N) + event.view(N, 1)) > 0

    comparable_pair = well_ordered_pair * lower_outcome_is_event

    out = all_pairs * comparable_pair.type(torch.uint8).cuda()

    return out.sum() / comparable_pair.sum()


def mse_loss(pred, obs, event):
    # for positive events, use vanilla MSE
    mse = event * ((pred - obs) ** 2)

    return mse.mean()


def penalty_loss(pred, obs, event):
    # for negative events (i.e. censored data points),
    # calculate MSE of events where pred < obs; error is
    # not defined when pred > obs
    p = (1 - event) * (pred < obs) * ((pred - obs) ** 2)
    return p.mean()


class DeepCENTLoss(torch.nn.Module):
    """DeepCENT loss function"""

    def __init__(self, lambda_p, lambda_r):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_r = lambda_r

    def forward(self, predictions, observations, events):
        return (
            mse_loss(predictions, observations, events),
            self.lambda_p * penalty_loss(predictions, observations, events),
            -self.lambda_r * lower_bound_rank_loss(predictions, observations, events),
        )

        #     for X_batch, y_batch, E_batch in test_loader:
        #         y_test_pred = model(X_batch)
        #         y_pred_list.append(y_test_pred.cpu().numpy())
        #         y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        #         y_pred_list = sum(y_pred_list, [])
        #     result.append(y_pred_list)

        # result = np.array(result)
        # y_test_pred_mean = result.mean(axis=0).reshape(
        #     -1,
        # )
        # y_test_pred_sd = result.std(axis=0).reshape(
        #     -1,
        # )
        # y_pred_list_upper = y_test_pred_mean + 1.96 * y_test_pred_sd
        # y_pred_list_lower = y_test_pred_mean - 1.96 * y_test_pred_sd

    # return y_pred_list0, y_pred_list, y_pred_list_upper, y_pred_list_lower
