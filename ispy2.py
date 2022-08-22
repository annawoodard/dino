import os
from monai.data import CacheDataset
from tkinter import W
import nibabel as nib
import urllib.request
import torch
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import tabulate

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from utils import cached_get_dicom, stratified_group_split

from tcia_download import download_collection, dicom_collection_to_nifti

logger = logging.getLogger()

series_map = {
    "unilateral_dce": "ISPY2: VOLSER: uni-lateral cropped: original DCE",
    "bilateral_dce": "ISPY2: VOLSER: bi-lateral: original DCE",
}
inverted_series_map = {
    "ISPY2: VOLSER: uni-lateral cropped: original DCE": "unilateral_dce",
    "ISPY2: VOLSER: bi-lateral: original DCE": "bilateral_dce",
}


def log_summary(label, df):
    table = [
        [
            len(df[df.pCR == 1]),
            df[df.pCR == 1].StudyInstanceUID.nunique(),
            df[df.pCR == 1].PatientID.nunique(),
            len(df[df.pCR == 0]),
            df[df.pCR == 0].StudyInstanceUID.nunique(),
            df[df.pCR == 0].PatientID.nunique(),
        ],
    ]
    headers = [
        "series",
        "studies",
        "unique women",
        "series",
        "studies",
        "unique women",
    ]
    table_text = tabulate.tabulate(table, headers)
    table_width = len(table_text.split("\n")[0])
    label = label + " dataset summary"
    label_width = len(label)
    padding = table_width - label_width - 1
    logger.info(
        f"\n{label} {'*' * padding}\n"
        + " " * 30
        + "pCR"
        + " " * 30
        + "non-pCR\n"
        + "_" * 35
        + "  "
        + "_" * 35
        + "\n"
        + table_text
        + "\n"
    )


class ISPY2Dataset(CacheDataset):
    def __init__(
        self,
        transform=None,
        exclude: pd.core.series.Series = None,
        # image_size: int = (2016, 3808),
        prescale: float = 1.0,
        debug: bool = False,
        data_path: str = "/net/projects/cdac/annawoodard/ispy2",
        metadata=None,
        include_series=None,
        require_series=None,
        timepoints=None,
        cache_num=24,
        cache_rate=1.0,
        num_workers=4,
    ):
        self.data_path = data_path
        if transform is None:
            transform = torch.nn.Identity()
        if metadata is None:
            self.metadata = self.preprocess()
        else:
            self.metadata = metadata
        self.metadata = self.metadata[self.metadata.nifti_exists == True]

        if exclude is not None:
            original = self.metadata.path.nunique()
            self.metadata = self.metadata[~self.metadata["PatientID"].isin(exclude)]
            print(
                f"dropped {original - metadata.path.nunique()} views from patients in the finetuning testing set"
            )
        if prescale:
            self.metadata = self.metadata.sample(frac=prescale)
        if debug:
            metadata = metadata[:16]
        if include_series is not None:
            self.metadata = self.metadata[~pd.isnull(self.metadata.SeriesDescription)]
            self.metadata = self.metadata[
                self.metadata.SeriesDescription.str.contains(
                    "|".join([series_map[x] for x in include_series])
                )
            ]
        if timepoints is not None:
            self.metadata = self.metadata[
                self.metadata.ClinicalTrialTimePointID.str.contains(
                    "|".join(timepoints)
                )
            ]
        if require_series is not None:
            require_series = set([series_map[x] for x in require_series])
            passing_studies = []
            for study in self.metadata.StudyInstanceUID.unique():
                series = set(
                    self.metadata[
                        self.metadata.StudyInstanceUID == study
                    ].SeriesDescription.to_list()
                )
                if len(require_series & series) >= len(require_series):
                    passing_studies.append(study)
            self.metadata = self.metadata[
                self.metadata.StudyInstanceUID.str.contains("|".join(passing_studies))
            ]
        start = len(self.metadata)
        self.metadata = self.metadata[self.metadata.nifti_exists == True]
        print(f"removing {start - len(self.metadata)} series with missing nifti files")

        studies = []
        for study in self.metadata.StudyInstanceUID.unique():
            series = self.metadata[self.metadata.StudyInstanceUID == study]
            data = {
                "PatientID": series.iloc[0].PatientID,
                "StudyInstanceUID": series.iloc[0].StudyInstanceUID,
                "ClinicalTrialTimePointID": series.iloc[0].ClinicalTrialTimePointID,
                "pcR": series.iloc[0].pCR,
            }
            for description, path in zip(series.SeriesDescription, series.nifti_path):
                data[inverted_series_map[description]] = path
            studies.append(data)

        log_summary("train + validation", self.metadata)
        super(ISPY2Dataset, self).__init__(
            data=studies,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def preprocess(self):
        if not os.path.isfile(
            os.path.join(self.data_path, "dicom", "per_series_metadata.csv")
        ):
            download_collection(
                "ISPY2",
                "MR",
                self.data_path,
            )
        if not os.path.isfile(
            os.path.join(self.data_path, "nifti", "per_series_metadata.csv")
        ):
            dicom_collection_to_nifti(
                os.path.join(self.data_path, "dicom"),
                os.path.join(self.data_path, "nifti"),
            )

        if not os.path.isfile(os.path.join(self.data_path, "metadata.csv")):
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70230072/ISPY2%20Imaging%20Cohort%201%20Clinical%20Data.xlsx?api=v2",
                os.path.join(self.data_path, "clinical_data.xlsx"),
            )
            clinical_data = pd.read_excel(
                os.path.join(self.data_path, "clinical_data.xlsx")
            )
            nifti_series_metadata = pd.read_csv(
                os.path.join(self.data_path, "nifti", "per_series_metadata.csv")
            )
            clinical_data = clinical_data.rename(columns={"Patient_ID": "PatientID"})

            # metadata now has one row per series; each row has pCR outcome
            metadata = nifti_series_metadata.join(
                clinical_data.set_index("PatientID"), on="PatientID"
            )

            metadata.to_csv(os.path.join(self.data_path, "metadata.csv"), index=False)
        metadata = pd.read_csv(os.path.join(self.data_path, "metadata.csv"))
        return metadata


def load_metadata(
    test_size=0.0,
    prescale=1.0,
    metadata_path="/net/projects/cdac/annawoodard/ispy2/metadata.csv",
):
    metadata = pd.read_csv(metadata_path)
    if prescale:
        print(f"will use prescale of {prescale}")
        metadata = metadata.sample(frac=prescale)

    if test_size > 0:
        fit_metadata, test_metadata = stratified_group_split(
            metadata, "PatientID", "pCR", test_size
        )
    else:
        fit_metadata = metadata
        test_metadata = pd.DataFrame(columns=metadata.columns)

    return fit_metadata, test_metadata


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
            metadata, "PatientID", "pCR", test_size
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
            ISPY2Dataset(
                metadata=metadata.iloc[train_indexes], transform=train_transform
            ),
            ISPY2Dataset(
                metadata=metadata.iloc[val_indexes],
                transform=val_transform,
            ),
        )
        for train_indexes, val_indexes in cv.split(
            fit_indices, fit_metadata.malignant, fit_metadata.ID1
        )
    ]

    test_dataset = ISPY2Dataset(
        metadata=test_metadata,
        transform=val_transform,
    )

    log_summary("train + validation", fit_metadata)
    log_summary("testing", test_metadata)

    return fit_datasets, test_dataset
