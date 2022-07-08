import logging
import time
import torch

logger = logging.getLogger()


# @torch.jit.script
def is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))


# @torch.jit.script
def is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = torch.Tensor([0.0]).cuda()
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j:
        if d_i & d_j:
            conc = 1.0 - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * is_comparable(t_i, t_j, d_i, d_j)


def one_pair(x0, x1):
    return 1 + nn.LogSigmoid()(x1 - x0) / np.log(2.0)


# @torch.jit.script
def sum_concordant_disc(predicted_scores, durations, events_observed):
    # count = torch.Tensor([0.0]).cuda()
    count = 0.0
    n = len(predicted_scores)
    for i in range(n):
        for j in range(n):
            if j != i:
                count = count + is_concordant(
                    predicted_scores[i],
                    predicted_scores[j],
                    durations[i],
                    durations[j],
                    events_observed[i],
                    events_observed[j],
                )
    return count


# @torch.jit.script
def sum_comparable(durations, events_observed):
    count = torch.Tensor([0.0]).cuda()
    n = len(durations)
    for i in range(n):
        for j in range(n):
            if j != i:
                count = count + is_comparable(
                    durations[i], durations[j], events_observed[i], events_observed[j]
                )
    return count


def calculate_lower_bound_rank_loss(pred, obs, event):
    N = pred.size(0)
    # N x N
    all_pairs = one_pair(pred.view(N, 1), pred.view(1, N))

    well_ordered_pair = (obs.view(1, N) - obs.view(N, 1)) > 0

    # True for pairs where Y_i < Y_j and Y_i is an event; False otherwise
    lower_outcome_is_event = (event.view(1, N) + event.view(N, 1)) > 0

    comparable_pair = well_ordered_pair * lower_outcome_is_event

    out = all_pairs * comparable_pair.type(torch.uint8).cuda()

    return out.sum() / comparable_pair.sum()


# TODO estimate censoring distribution
def td_concordance_index(durations, predicted_scores, events_observed=None):
    """Time dependent concorance index from
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    We have made a small modifications
    for ties in predictions and event times.
    We have followed step 3. in Sec 5.1. in Random Survial Forests paper, except for the last
    point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
    See '_is_concordant'.

    This is modified for pytorch and predicted scores rather than survival function
    from the original pycox implementation:
    Håvard Kvamme and Ørnulf Borgan. Continuous and discrete-time survival prediction with neural networks. arXiv preprint arXiv:1910.06724, 2019. [paper]

    Arguments:
        durations {np.array[n]} -- Event times (or censoring times.)
        events {np.array[n]} -- Event indicators (0 is censoring).
        surv {np.array[n_times, n]} -- Survival function (each row is a duraratoin, and each col
            is an individual).

    Returns:
        float -- Time dependent concordance index.
    """
    assert durations.shape[0] == predicted_scores.shape[0] == events_observed.shape[0]
    assert (
        type(durations)
        is type(events_observed)
        is type(predicted_scores)
        is torch.Tensor
    )
    # if events.dtype in ("float", "float32"):
    #     events = events.astype("int32")
    # durations = durations.astype("float")

    try:
        return sum_concordant_disc(
            predicted_scores, durations, events_observed
        ) / sum_comparable(durations, events_observed)
    except ZeroDivisionError:
        print(
            "The c-index is not well-defined if there are no positive examples; returning c-index=-1"
        )
        return -1.0
