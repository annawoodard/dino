import torch


# TODO estimate censoring distribution
# @torch.jit.script
def td_concordance_index(pred, obs, event):
    """Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    We have made a small modifications
    for ties in predictions and event times.
    We have followed step 3. in Sec 5.1. in Random Survial Forests paper, except for the last
    point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.

    This is modified (numpy -> pytorch and survival function -> predicted time-to-event)
    from the original pycox implementation:
    Håvard Kvamme and Ørnulf Borgan. Continuous and discrete-time survival prediction with neural networks. arXiv preprint arXiv:1910.06724, 2019.

    Arguments:
        obs {torch.Tensor[n]} -- Event times (or censoring times.)
        event {torch.Tensor[n]} -- Event indicators (0 is censoring).
        pred {torch.Tensor[n_times, n]} -- Predicted time-to-event (each row is a duration, and each col
            is an individual).

    Returns:
        float -- Time dependent concordance index.
    """
    assert obs.shape[0] == pred.shape[0] == event.shape[0]
    assert type(obs) is type(pred) is type(event) is torch.Tensor

    N = len(pred)
    concordant = torch.zeros((N, N)).cuda()
    ones = torch.ones((N, N)).cuda()
    obs_i_lt_obs_j = obs.view(1, N) < obs.view(N, 1)
    obs_i_eq_obs_j = obs.view(1, N) == obs.view(N, 1)
    pred_i_lt_pred_j = pred.view(1, N) < pred.view(N, 1)
    pred_i_gt_pred_j = pred.view(1, N) > pred.view(N, 1)
    pred_i_eq_pred_j = pred.view(1, N) == pred.view(N, 1)
    pred_i_neq_pred_j = pred.view(1, N) != pred.view(N, 1)
    event_i_and_event_j = event.bool().view(1, N) & event.bool().view(N, 1)
    event_i = event.bool().view(1, N)
    event_j = event.bool().view(N, 1)

    concordant[obs_i_lt_obs_j] = (
        pred_i_lt_pred_j.type(torch.uint8)[obs_i_lt_obs_j]
        + pred_i_eq_pred_j.type(torch.uint8)[obs_i_lt_obs_j] * 0.5
    )
    concordant[obs_i_eq_obs_j & event_i_and_event_j] = (
        ones[obs_i_eq_obs_j & event_i_and_event_j]
        - pred_i_neq_pred_j.type(torch.uint8)[obs_i_eq_obs_j & event_i_and_event_j]
        * 0.5
    )
    concordant[obs_i_eq_obs_j & event_i] = (
        pred_i_lt_pred_j.type(torch.uint8)[obs_i_eq_obs_j & event_i]
        + pred_i_eq_pred_j.type(torch.uint8)[obs_i_eq_obs_j & event_i] * 0.5
    )
    concordant[obs_i_eq_obs_j & event_j] = (
        pred_i_gt_pred_j.type(torch.uint8)[obs_i_eq_obs_j & event_j]
        + pred_i_eq_pred_j.type(torch.uint8)[obs_i_eq_obs_j & event_j] * 0.5
    )

    # True for pairs where Y_i < Y_j and Y_i is an event; False otherwise
    # lower_outcome_is_event = (event.view(1, N) + event.view(N, 1)) > 0
    # well_ordered_pair = (obs.view(1, N) - obs.view(N, 1)) > 0
    lower_outcome_is_event = (event.view(N, 1) + event.view(1, N)) > 0
    well_ordered_pair = (obs.view(N, 1) - obs.view(1, N)) > 0
    comparable_pair = well_ordered_pair * lower_outcome_is_event

    out = concordant * comparable_pair.type(torch.uint8)

    try:
        return out.sum() / comparable_pair.sum()
    except ZeroDivisionError:
        print(
            "The c-index is not well-defined if there are no comparable pairs; returning c-index=-1"
        )
        return -1.0
