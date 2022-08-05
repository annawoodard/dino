import os
import urllib.request
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from retry import retry
import tabulate

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from utils import cached_get_dicom, stratified_group_split

from tcia_download import download_collection

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
        data_path: str = "/net/scratch/annawoodard/cmmd",
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
        metadata_path = os.path.join(self.data_path, "metadata.csv")
        if os.path.isfile(metadata_path):
            return pd.read_csv(metadata_path)

        urllib.request.urlretrieve(
            "https://wiki.cancerimagingarchive.net/download/attachments/70230508/CMMD_clinicaldata_revision.xlsx?api=v2",
            os.path.join(self.data_path, "clinical_data.xlsx"),
        )
        download_collection(
            "CMMD",
            "MG",
            self.data_path,
            save_tags=[
                ("ImageLaterality", "ImageLaterality"),
                (lambda x: x.ViewCodeSequence[0].CodeMeaning, "view"),
            ],
        )
        clinical_data = pd.read_excel(
            os.path.join(self.data_path, "clinical_data.xlsx")
        )
        metadata = pd.read_csv(metadata_path)
        clinical_data["PatientID"] = clinical_data.ID1
        metadata = metadata.join(clinical_data.set_index("PatientID"), on="PatientID")

        metadata["malignant"] = (metadata.LeftRight == metadata.ImageLaterality) & (
            metadata.classification == "Malignant"
        )
        metadata.malignant = metadata.malignant.astype(int)
        metadata.to_csv(metadata_path, index=False)
        return metadata

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
    metadata_path="",
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


def background_crop(image):
    # count number of white pixels in columns as new 1D array
    count_cols = torch.count_nonzero(image, dim=1)

    # get first and last x coordinate where black
    first_x = torch.argwhere(count_cols.squeeze() > 0)[0].item()
    last_x = torch.argwhere(count_cols.squeeze() > 0)[-1].item()

    return image[
        :,
        :,
        first_x : last_x + 1,
    ]


class CMMDRandomTileDataset(CMMDDataset):
    def __init__(
        self,
        transform=None,
        exclude: pd.core.series.Series = None,
        prescale: float = 1.0,
        debug: bool = False,
        data_path: str = "/net/scratch/annawoodard/cmmd",
        metadata=None,
        tiles_per_view=20,
        tile_size=224,
        max_frac_black=0.96,
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
        self.max_frac_black = max_frac_black
        self.crop = transforms.RandomCrop(tile_size)

    @retry(tries=10)
    def get_tile(self, image):
        # tile_pil = self.crop(image)
        bg_cropped_tile = background_crop(F.to_tensor(image))

        frac_black = (
            bg_cropped_tile[bg_cropped_tile == bg_cropped_tile.max()].view(-1).shape[0]
            / bg_cropped_tile.contiguous().view(-1).shape[0]
        )
        if frac_black < self.max_frac_black:
            return F.to_pil_image(bg_cropped_tile)
        raise RuntimeError("Couldn't find non-background tile after ten attempts")

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
        tile = self.get_tile(cached_get_dicom(view.path))
        return self.transform(tile)
        # return self.transform(cached_get_dicom(view.path))
