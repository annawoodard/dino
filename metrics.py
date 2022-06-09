from pathlib import Path
import pandas as pd
import time
from sklearn import metrics

import torch

from pycox.models.coxph import compute_baseline_hazards, output2surv
from pycox.evaluation import EvalSurv


def predict_coxph_surv(
    train_outputs,
    train_durations,
    train_events,
    test_outputs,
    output_dir=".",
    sample=1.0,
):
    """Mostly copy and paste from pycox. See
    https://github.com/havakv/pycox/blob/refactor_out_torchtuples/pycox/models/coxph.py
    """
    baseline_hazards, durations = compute_baseline_hazards(
        train_outputs,
        train_durations,
        train_events,
    )

    cumulative_baseline_hazards = baseline_hazards.cumsum(0)

    surv = output2surv(test_outputs, cumulative_baseline_hazards)
    surv = pd.DataFrame(surv.transpose(0, 1).cpu().numpy(), durations.cpu().numpy())
    surv.to_pickle(Path(output_dir) / "surv.pkl")

    return surv


def concordance_index(surv, durations, events):
    durations = durations.cpu().numpy()
    events = events.cpu().numpy()
    try:
        start = time.time()
        ev = EvalSurv(surv, durations, events, censor_surv="km")
        c_index = ev.concordance_td()
        print(
            "Computed c-index on {} samples in {:.0f}s".format(
                events.shape[0], time.time() - start
            )
        )

        return c_index
    except ZeroDivisionError:
        print(
            "The c-index is not well-defined if there are no positive examples; returning c-index=-1"
        )
        return -1


def auc(surv, durations, events, years=(1, 2, 5)):
    result = {}
    durations = durations.cpu().numpy()
    events = events.cpu().numpy()
    for year in years:
        y = durations.copy()
        y[durations > year] = 0
        y[(durations <= year) & (events == 0)] = 0
        y[(durations <= year) & (events == 1)] = 1

        pred = surv[surv.index > year].iloc[0]
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(
            f"year {year} <===> {len(y[y==0])} controls and {len(y[y==1])} cases <===> auc: {auc:.2f}"
        )
        result["{year}_year_auc"] = auc

    return result
