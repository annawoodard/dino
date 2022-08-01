import os
import numpy as np
import glob
import pydicom
import itertools
from tqdm import tqdm
import torch.distributed as dist
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from retry import retry
import tabulate

# import cv2
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.models.layers import to_2tuple
from torch.utils.data import Dataset
from torchvision import transforms
from monai.transforms.spatial.array import GridPatch
from utils import get_dicom
from utils import stratified_group_split

logger = logging.getLogger()


def log_summary(label, df):
    table = [
        [
            len(df[df.malignant == 1]),
            df[df.malignant == 1].ID1.nunique(),
            len(df[df.malignant == 0]),
            df[df.malignant == 0].ID1.nunique(),
        ],
    ]
    headers = [
        "views",
        "unique women",
        "views",
        "unique women",
    ]
    table_text = tabulate.tabulate(table, headers)
    table_width = len(table_text.split("\n")[0])
    label = label + " dataset summary"
    label_width = len(label)
    padding = table_width - label_width - 1
    logger.info(
        f"\n{label} {'*' * padding}\n"
        + " " * 15
        + "malignant"
        + " " * 25
        + "benign\n"
        + "_" * 32
        + "  "
        + "_" * 32
        + "\n"
        + table_text
        + "\n"
    )


class CMMDDataset(Dataset):
    def __init__(
        self,
        transform,
        exclude: pd.core.series.Series = None,
        # image_size: int = (2016, 3808),
        prescale: float = 1.0,
        debug: bool = False,
        data_path: str = "/gpfs/data/huo-lab/Image/CMMD",
        metadata=None,
        **pad_kwargs,
    ):
        self.data_path = data_path
        if metadata is None:
            self.metadata = self.preprocess()
        else:
            self.metadata = metadata

        if exclude is not None:
            original = self.metadata.path.nunique()
            self.metadata = self.metadata[~self.metadata["ID1"].isin(exclude)]
            print(
                f"dropped {original - metadata.path.nunique()} views from patients in the finetuning testing set"
            )
        if prescale:
            self.metadata = self.metadata.sample(frac=prescale)
        if debug:
            metadata = metadata[:16]
        # self.image_size = to_2tuple(image_size)
        self.transform = transform
        self.load_views()

    def load_views(self):
        start = time.time()
        self.views = []
        for fn in tqdm(self.metadata.path):
            self.views.append(get_dicom(fn))
        logger.info(
            "finished loading {} views in {:.0f}s".format(
                len(self), time.time() - start
            )
        )

    def preprocess(self):
        path = os.path.join(self.data_path, "per_view_metadata.csv")
        if os.path.isfile(path):
            return pd.read_csv(path)

        cmmd = pd.read_excel(
            os.path.join(self.data_path, "CMMD_clinicaldata_revision.xlsx")
        )
        data = []
        dicoms = glob.glob(os.path.join(self.data_path, "*/*/*/*dcm"))
        for path in tqdm(dicoms):
            f = pydicom.dcmread(path)
            row = cmmd[cmmd.ID1 == f.PatientID].to_dict("records")[0].copy()
            row["ImageLaterality"] = f.ImageLaterality
            row["path"] = path
            row["view"] = f.ViewCodeSequence[0].CodeMeaning
            data.append(row)

        data = pd.DataFrame(data)
        data["malignant"] = (data.LeftRight == data.ImageLaterality) & (
            data.classification == "Malignant"
        )
        data.malignant = data.malignant.astype(int)
        data.exam_id
        data.to_csv(os.path.join(self.data_path, "per_view_metadata.csv"))
        return data

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple:
        """Get item.

        Args:
            idx (int): Index in the metadata table to retrieve.

        Returns:
            Tuple of images:
        """
        view = self.metadata.iloc[index]
        # image = self.transform(get_dicom(view.path))
        image = self.transform(self.views[index])

        # frac_black = result[0][result[0] < 0.0].shape[0] / result[0].view(-1).shape[0]
        # while frac_black > self.max_frac_black:
        #     result = self.transform(self.crop(image))
        #     frac_black = (
        #         result[0][result[0] < 0.0].shape[0] / result[0].view(-1).shape[0]
        #     )
        return image, view.malignant


def get_datasets(
    prescale,
    train_transform,
    val_transform,
    n_splits,
    random_state,
    test_size=0.15,
    metadata_path="/gpfs/data/huo-lab/Image/CMMD/per_view_metadata.csv",
):
    metadata = pd.read_csv(metadata_path)
    if prescale:
        print(f"will use prescale of {prescale}")
        metadata = metadata.sample(frac=prescale)

    if test_size > 0:
        fit_metadata, test_metadata = stratified_group_split(
            metadata, "ID1", "malignant", test_size
        )
    else:
        fit_metadata = metadata
        test_metadata = pd.DataFrame(columns=metadata.columns)

    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    indexes = np.arange(len(metadata))
    fit_datasets = [
        (
            CMMDDataset(metadata.iloc[train_indexes], image_transform=train_transform),
            CMMDDataset(
                metadata.iloc[val_indexes],
                image_transform=val_transform,
            ),
        )
        for train_indexes, val_indexes in cv.split(
            indexes, fit_metadata.malignant, fit_metadata.ID1
        )
    ]

    test_dataset = CMMDDataset(
        test_metadata,
        image_transform=val_transform,
    )

    log_summary("train + validation", fit_metadata)
    log_summary("testing", test_metadata)

    return fit_datasets, test_dataset
