import argparse
import os
import pandas as pd
import tabulate

parser = argparse.ArgumentParser("Print finetuning results")
parser.add_argument(
    "--labels",
    type=str,
    nargs="+",
    help="label for result",
)
parser.add_argument(
    "--dirs",
    type=str,
    nargs="+",
    help="dirs with results",
)
args = parser.parse_args()

table = []
headers = ["model", "num folds", "c-index (std)", "test c-index"]

for label, directory in zip(args.labels, args.dirs):
    results = pd.read_pickle(os.path.join(directory, "results.pkl"))
    table += [
        [
            label,
            str(len(results) - 1),
            "{:.3f} ({:.3f})".format(results.c_index.mean(), results.c_index.std()),
            "{:.3f}".format(results[pd.isnull(results.fold)].c_index.iloc[0]),
        ]
    ]

print(tabulate.tabulate(table, headers))
