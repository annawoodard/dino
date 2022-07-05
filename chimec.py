import os
import logging
import time
from typing import Callable, Dict, Optional, Tuple, Union

# import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import tabulate
import torch
from maicara.preprocessing.utils import EmptyCrop
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.models.layers import to_2tuple
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger()


def stratified_group_split(
    # adapted from https://stackoverflow.com/a/64663756/5111510
    samples: pd.DataFrame,
    group: str,
    stratify_by: str,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = samples[group].drop_duplicates()
    stratify = samples.drop_duplicates(group)[stratify_by].to_numpy()
    groups_train, groups_test = train_test_split(
        groups, stratify=stratify, test_size=test_size
    )

    samples_train = samples.loc[lambda d: d[group].isin(groups_train)]
    samples_test = samples.loc[lambda d: d[group].isin(groups_test)]

    return samples_train, samples_test


def worker_init_fn(worker_id):
    """Handle random seeding."""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2**32 - 1)

    np.random.seed(seed)


def log_summary(label, df):
    table = [
        [
            len(df[df.event == 1]),
            df[df.event == 1].exam_id.nunique(),
            df[df.event == 1].study_id.nunique(),
            len(df[df.event == 0]),
            df[df.event == 0].exam_id.nunique(),
            df[df.event == 0].study_id.nunique(),
        ],
    ]
    headers = [
        "views",
        "exams",
        "unique women",
        "views",
        "exams",
        "unique women",
    ]
    table_text = tabulate.tabulate(table, headers)
    table_width = len(table_text.split("\n")[0])
    censoring_proportion = len(df[df.event == 0]) / len(df) * 100
    label = (
        label + f" dataset summary (censoring proportion: {censoring_proportion:.0f}%)"
    )
    label_width = len(label)
    padding = table_width - label_width - 1
    logger.info(f"\n{label} {'*' * padding}")
    logger.info(" " * 15 + "cases" + " " * 25 + "controls")
    logger.info("_" * 32 + "  " + "_" * 32)
    logger.info(table_text + "\n")


class ChiMECSSLDataset(Dataset):
    def __init__(
        self,
        transform,
        exclude: pd.core.series.Series = None,
        image_size: int = 224,
        prescale: float = 1.0,
    ):
        metadata = pd.read_pickle(
            "/gpfs/data/huo-lab/Image/annawoodard/maicara/data/interim/mammo_loose_cuts_v3/series_metadata.pkl"
        )
        # metadata passing all filters will have `np.nan` in the `filter` column
        metadata = metadata[
            (pd.isnull(metadata["filter"])) & (~pd.isnull(metadata["png_path"]))
        ]
        if exclude is not None:
            original = metadata.exam_id.nunique()
            metadata = metadata[~metadata["study_id"].isin(exclude)]
            print(
                f"dropped {original - metadata.exam_id.nunique()} exams from patients in the finetuning testing set"
            )
        if prescale:
            # need to ensure we get some cases, even for small prescale values
            cases = metadata[metadata.event == 1].sample(frac=min(prescale * 3, 1))
            controls = metadata[metadata.event == 0].sample(frac=prescale)
            metadata = pd.concat([cases, controls]).sample(frac=1)
        self.metadata = metadata
        log_summary("pretraining mammo\n\n", self.metadata)
        # TODO ensure all inputs are in same orientation
        # TODO fix aspect ratio-- unilateral images should have final size h, h/2.
        self.resize = transforms.Resize(to_2tuple(image_size))
        self.transform = transform

    def load_png(self, filename):
        try:
            image = Image.open(filename).convert("L")
        except OSError:
            time.sleep(2)
            image = Image.open(filename).convert("L")
        return image

    def __len__(self) -> int:
        length = 0
        if self.metadata is not None:
            length = len(self.metadata)

        return length

    def __getitem__(self, idx: int) -> Tuple:
        """Get item.

        Args:
            idx (int): Index in the metadata table to retrieve.

        Returns:
            Tuple of images:
        """
        series = self.metadata.iloc[idx]
        filename = series["png_path"]
        image = self.resize(self.load_png(filename))
        return self.transform(image)


class ChiMECStackedSSLDataset(Dataset):
    def __init__(
        self,
        transform,
        exclude: pd.core.series.Series = None,
        image_size: int = 224,
        prescale: float = 1.0,
    ):
        metadata = pd.read_pickle(
            "/gpfs/data/huo-lab/Image/annawoodard/maicara/data/interim/mammo_loose_cuts_v3/series_metadata.pkl"
        )
        # metadata passing all filters will have `np.nan` in the `filter` column
        metadata = metadata[
            (pd.isnull(metadata["filter"])) & (~pd.isnull(metadata["png_path"]))
        ]
        if exclude is not None:
            original = metadata.exam_id.nunique()
            metadata = metadata[~metadata["study_id"].isin(exclude)]
            print(
                f"dropped {original - metadata.exam_id.nunique()} exams from patients in the finetuning testing set"
            )

        if prescale:
            # need to ensure we get some cases, even for small prescale values
            cases = metadata[metadata.event == 1].sample(frac=min(prescale * 3, 1))
            controls = metadata[metadata.event == 0].sample(frac=prescale)
            metadata = pd.concat([cases, controls]).sample(frac=1)

        metadata["lateral_view_set"] = (
            metadata["exam_id"] + " " + metadata["ImageLaterality"]
        )
        self.metadata = metadata
        log_summary("pretraining mammo\n\n", self.metadata)
        # TODO ensure all inputs are in same orientation
        # TODO fix aspect ratio-- unilateral images should have final size h, h/2.
        self.resize = transforms.Resize(to_2tuple(image_size))
        self.transform = transform
        self.lateral_view_sets = set(metadata.lateral_view_set.unique())
        original = len(self.lateral_view_sets)
        self.lateral_view_sets = []
        missing_views = 0
        for view_set in metadata.lateral_view_set.unique():
            views = metadata[
                metadata.lateral_view_set == view_set
            ].ViewPosition.tolist()
            if ("MLO" in views) and ("CC" in views):
                self.lateral_view_sets += [view_set]
            else:
                missing_views += 1
        # TODO investigate, these should already be removed
        print(f"Removed {missing_views} view sets without both views")

    def load_png(self, filename):
        try:
            image = Image.open(filename).convert("L")
        except OSError:
            time.sleep(2)
            image = Image.open(filename).convert("L")
        return image

    def __len__(self) -> int:
        if self.metadata is not None:
            return len(self.lateral_view_sets)
        return 0

    def __getitem__(self, idx: int) -> Tuple:
        """Get item.

        Args:
            idx (int): Index in the metadata table to retrieve.

        Returns:
            Tuple of images:
        """
        lateral_view_set = self.lateral_view_sets[idx]
        cc_view = self.resize(
            self.load_png(
                self.metadata[
                    (self.metadata.lateral_view_set == lateral_view_set)
                    & (self.metadata.ViewPosition == "CC")
                ]
                .sample(1)
                .png_path.item()
            )
        )
        mlo_view = self.resize(
            self.load_png(
                self.metadata[
                    (self.metadata.lateral_view_set == lateral_view_set)
                    & (self.metadata.ViewPosition == "MLO")
                ]
                .sample(1)
                .png_path.item()
            )
        )
        cc_views = self.transform(cc_view)
        mlo_views = self.transform(mlo_view)
        res = [torch.cat((x1, x2)) for x1, x2 in zip(cc_views, mlo_views)]
        return res


class ChiMECFinetuningTrainingDataset(Dataset):
    def __init__(
        self,
        metadata: Union[str, os.PathLike],
        image_transform: Optional[Callable] = None,
    ):
        self.metadata = metadata
        self.image_transform = image_transform
        original = len(self.metadata.exam_id.unique())
        # FIXME sometimes additional views are coded as different exams, need to select on time
        # rather than exam ID if we are losing many exams
        self.exams = []
        for exam in metadata.exam_id.unique():
            lateralities = metadata[metadata.exam_id == exam].ImageLaterality.tolist()
            if set(["L", "R"]) == set(lateralities):
                self.exams += [exam]
        if (original - len(self.exams)) > 0:
            print(
                f"Removed {original - len(self.exams)} exams without both lateralities"
            )

    def load_image(self, filename):
        image = Image.open(filename).convert("L")
        if self.image_transform is not None:
            return self.image_transform(image)
        return image

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        series = self.metadata.iloc[idx]
        filename = series["png_path"]
        image = self.load_image(filename)
        contralateral_views = self.metadata[
            (self.metadata.study_id == series.study_id)
            & (self.metadata.exam_id == series.exam_id)
            & (self.metadata.ImageLaterality != series.ImageLaterality)
        ]
        if len(contralateral_views) > 0:
            contralateral_filename = contralateral_views.sample(1).png_path.item()
            contralateral_image = self.load_image(contralateral_filename)
        else:
            contralateral_image = image.clone()

        return (
            image,
            contralateral_image,
            series["years_to_event"],
            series["event"],
            series["study_id"],
        )


class ChiMECFinetuningEvalDataset(Dataset):
    def __init__(
        self,
        metadata: Union[str, os.PathLike],
        image_transform: Optional[Callable] = None,
    ):
        self.metadata = metadata
        self.image_transform = image_transform
        original = len(self.metadata.exam_id.unique())
        # FIXME sometimes additional views are coded as different exams, need to select on time
        # rather than exam ID if we are losing many exams
        self.exams = []
        for exam in metadata.exam_id.unique():
            lateralities = metadata[metadata.exam_id == exam].ImageLaterality.tolist()
            cc_views = metadata[
                (metadata.exam_id == exam) & (metadata.ViewPosition == "CC")
            ].ViewPosition.tolist()
            if (set(["L", "R"]) == set(lateralities)) and (len(cc_views) >= 2):
                self.exams += [exam]
        if (original - len(self.exams)) > 0:
            print(
                f"Removed {original - len(self.exams)} exams without both lateralities and CC views"
            )

    def load_image(self, filename):
        image = Image.open(filename).convert("L")
        if self.image_transform is not None:
            return self.image_transform(image)
        return image

    def __len__(self) -> int:
        return len(self.exams)

    def __getitem__(self, idx: int) -> Dict:
        exam = self.exams[idx]
        left_row = self.metadata[
            (self.metadata.exam_id == exam)
            & (self.metadata.ViewPosition == "CC")
            & (self.metadata.ImageLaterality == "L")
        ].sample(1)
        right_row = self.metadata[
            (self.metadata.exam_id == exam)
            & (self.metadata.ViewPosition == "CC")
            & (self.metadata.ImageLaterality == "R")
        ].sample(1)
        l_cc_view = self.load_image(left_row.png_path.item())
        r_cc_view = self.load_image(right_row.png_path.item())

        return (
            l_cc_view,
            r_cc_view,
            left_row["years_to_event"].values[0],
            left_row["event"].values[0],
            left_row["study_id"].values[0],
        )


class ChiMECStackedFinetuningDataset(Dataset):
    def __init__(
        self,
        metadata: Union[str, os.PathLike],
        image_transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.metadata = metadata
        self.image_transform = image_transform
        # TODO ensure all inputs are in same orientation
        # TODO fix aspect ratio-- unilateral images should have final size h, h/2.
        self.resize = transforms.Resize(to_2tuple(image_size))
        original = len(self.metadata.exam_id.unique())
        self.exams = []
        # FIXME sometimes additional views are coded as different exams, need to select on time
        # rather than exam ID if we are losing many exams
        for exam in metadata.exam_id.unique():
            left_views = metadata[
                (metadata.exam_id == exam) & (metadata.ImageLaterality == "L")
            ].ViewPosition.tolist()
            right_views = metadata[
                (metadata.exam_id == exam) & (metadata.ImageLaterality == "R")
            ].ViewPosition.tolist()
            if (
                ("MLO" in left_views)
                and ("CC" in left_views)
                and ("MLO" in right_views)
                and ("CC" in right_views)
            ):
                self.exams += [exam]
        if (original - len(self.exams)) > 0:
            print(
                f"Removed {original - len(self.exams)} exams without all four views (L MLO, L CC, R MLO, R CC)"
            )

    def load_image(self, filename):
        image = Image.open(filename).convert("L")
        if self.image_transform is not None:
            return self.image_transform(image)
        return image

    def __len__(self) -> int:
        return len(self.exams)

    def __getitem__(self, idx: int) -> Dict:
        exam = self.exams[idx]
        stacked_views = {}
        for laterality in ("L", "R"):
            try:
                cc_view = self.resize(
                    self.load_image(
                        self.metadata[
                            (self.metadata.exam_id == exam)
                            & (self.metadata.ViewPosition == "CC")
                            & (self.metadata.ImageLaterality == laterality)
                        ]
                        .sample(1)
                        .png_path.item()
                    )
                )
            except ValueError:
                raise ValueError(
                    f"Problem finding {laterality} CC view for exam {exam}"
                )
            try:
                mlo_view = self.resize(
                    self.load_image(
                        self.metadata[
                            (self.metadata.exam_id == exam)
                            & (self.metadata.ViewPosition == "MLO")
                            & (self.metadata.ImageLaterality == laterality)
                        ]
                        .sample(1)
                        .png_path.item()
                    )
                )
            except ValueError:
                raise ValueError(
                    f"Problem finding {laterality} MLO view for exam {exam}"
                )
            stacked_views[laterality] = torch.cat((cc_view, mlo_view))
        series = self.metadata[self.metadata.exam_id == exam].iloc[0]

        return (
            stacked_views["L"],
            stacked_views["R"],
            series["years_to_event"],
            series["event"],
            series["study_id"],
        )


class OversamplingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, oversampling_size=1000):
        self.dataset = dataset

        self.class_idx_to_sample_ids = {
            0: dataset.metadata[dataset.metadata.event == 0].index.to_list(),
            1: dataset.metadata[dataset.metadata.event == 1].index.to_list(),
        }

    def __len__(self):
        return self.num_classes * self.oversampling_size

    def __getitem__(self, index):
        sample_idx = random.sample(self.class_idx_to_sample_ids[class_id], 1)
        return self.dataset[sample_idx[0]]


def load_metadata(
    test_size,
    metadata_path="/gpfs/data/huo-lab/Image/annawoodard/maicara/data/interim/mammo_v10/clean_series_metadata.pkl",
):
    metadata = pd.read_pickle(metadata_path)
    # metadata passing all filters will have `np.nan` in the `filter` column
    metadata = metadata[
        (pd.isnull(metadata["filter"])) & (~pd.isnull(metadata["png_path"]))
    ]
    metadata["years_to_event"] = (
        pd.to_timedelta(metadata.time_to_event).dt.days / 365.25
    )

    if test_size > 0:
        fit_metadata, test_metadata = stratified_group_split(
            metadata, "study_id", "event", test_size
        )
    else:
        fit_metadata = metadata
        test_metadata = pd.DataFrame(columns=metadata.columns)

    return fit_metadata, test_metadata


# start = time.time()
# metadata["png_exists"] = metadata.png_path.apply(lambda x: os.path.isfile(x))
# metadata = metadata[metadata.png_exists == True]
# print("finished checking png paths in {:.0f}s".format(time.time() - start))
# metadata.to_pickle("/gpfs/data/huo-lab/Image/annawoodard/mae/metadata.pkl")


def get_datasets(
    prescale,
    train_transform,
    val_transform,
    test_metadata,
    fit_metadata,
    n_splits,
    random_state,
):
    if prescale:
        print(f"will use prescale of {prescale}")
        # need to ensure we get some cases, even for small prescale values
        cases = fit_metadata[fit_metadata.event == 1].sample(frac=min(prescale * 3, 1))
        controls = fit_metadata[fit_metadata.event == 0].sample(frac=prescale)
        fit_metadata = pd.concat([cases, controls]).sample(frac=1)

    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    indexes = np.arange(len(fit_metadata))

    fit_datasets = [
        (
            ChiMECFinetuningTrainingDataset(
                fit_metadata.iloc[train_indexes], image_transform=train_transform
            ),
            ChiMECFinetuningEvalDataset(
                fit_metadata.iloc[val_indexes],
                image_transform=val_transform,
            ),
        )
        for train_indexes, val_indexes in cv.split(
            indexes, fit_metadata.event, fit_metadata.study_id
        )
    ]

    test_dataset = ChiMECFinetuningEvalDataset(
        test_metadata,
        image_transform=val_transform,
    )

    log_summary("train + validation", fit_metadata)
    log_summary("testing", test_metadata)

    # # Required by cumulative_dynamic_auc from sksurv.metrics for estimating
    # # the censorship distribution.
    # survival_train = np.array(
    #     [tuple(x) for x in metadata[["event", "duration"]].values],
    #     dtype="?, f8",
    # )

    # self.in_features = self.train_dataset[0][0].shape[1]

    return fit_datasets, test_dataset
