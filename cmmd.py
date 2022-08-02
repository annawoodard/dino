import os
import torchvision.transforms.functional as F
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
from utils import cached_get_dicom
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
        + " " * 10
        + "malignant"
        + " " * 20
        + "benign\n"
        + "_" * 23
        + "  "
        + "_" * 23
        + "\n"
        + table_text
        + "\n"
    )


class CMMDDataset(Dataset):
    def __init__(
        self,
        transform=None,
        exclude: pd.core.series.Series = None,
        # image_size: int = (2016, 3808),
        prescale: float = 1.0,
        debug: bool = False,
        data_path: str = "/gpfs/data/huo-lab/Image/CMMD",
        metadata=None,
        return_label=True,
    ):
        self.data_path = data_path
        if transform is None:
            transform = torch.nn.Identity()
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
        self.return_label = return_label
        log_summary("train + validation", self.metadata)

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
        data.to_csv(os.path.join(self.data_path, "per_view_metadata.csv"), index=False)
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
        # this is a small dataset; we can cache full copies on all workers
        image = self.transform(cached_get_dicom(view.path))
        if self.return_label:
            return image, view.malignant
        else:
            return image


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
    fit_indices = np.arange(len(fit_metadata))
    fit_datasets = [
        (
            CMMDDataset(
                metadata=metadata.iloc[train_indexes], transform=train_transform
            ),
            CMMDDataset(
                metadata=metadata.iloc[val_indexes],
                transform=val_transform,
            ),
        )
        for train_indexes, val_indexes in cv.split(
            fit_indices, fit_metadata.malignant, fit_metadata.ID1
        )
    ]

    test_dataset = CMMDDataset(
        metadata=test_metadata,
        transform=val_transform,
    )

    log_summary("train + validation", fit_metadata)
    log_summary("testing", test_metadata)

    return fit_datasets, test_dataset


@retry(tries=10)
def get_tile(crop, image, max_frac_black):
    tile_pil = crop(image)
    tile_tensor = F.to_tensor(tile_pil)
    frac_black = (
        tile_tensor[tile_tensor == tile_tensor.max()].view(-1).shape[0]
        / tile_tensor.view(-1).shape[0]
    )
    if frac_black < max_frac_black:
        return tile_pil
    raise RuntimeError("Couldn't find non-background tile after ten attempts")


class CMMDRandomTileDataset(CMMDDataset):
    def __init__(
        self,
        transform=None,
        exclude: pd.core.series.Series = None,
        prescale: float = 1.0,
        debug: bool = False,
        data_path: str = "/gpfs/data/huo-lab/Image/CMMD",
        metadata=None,
        tiles_per_view=20,
    ):
        super().__init__(
            transform=transform,
            exclude=exclude,
            prescale=prescale,
            debug=debug,
            data_path=data_path,
            metadata=metadata,
        )
        self.tiles_per_view = tiles_per_view

    def __len__(self) -> int:
        return len(self.metadata) * self.tiles_per_view

    def __getitem__(self, index: int) -> Tuple:
        """Get item.

        Args:
            idx (int): Index in the metadata table to retrieve.

        Returns:
            Tuple of images:
        """
        view = self.metadata.iloc[index % len(self.metadata)]
        # this is a small dataset; we can cache full copies on all workers
        return self.transform(cached_get_dicom(view.path))
