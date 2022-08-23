from genericpath import isfile
import os
import shutil
import time
import glob
import pydicom
from tciaclient.core import TCIAClient
import pandas as pd
import subprocess
from tqdm import tqdm
import argparse
from p_tqdm import p_umap
import dicom2nifti
import dicom2nifti.settings as settings

settings.disable_validate_slice_increment()
settings.disable_validate_slicecount()


def download_series(series, download_path, save_tags):
    series = series.copy()
    series_dir = os.path.join(
        download_path,
        series["PatientID"],
        series["StudyInstanceUID"],
        series["SeriesInstanceUID"],
    )
    series_partial_dir = series_dir + ".partial"
    if os.path.isdir(series_partial_dir):
        shutil.rmtree(series_partial_dir)  # cleanup previous download attempt
    if os.path.isdir(series_dir):
        if len(os.listdir(series_dir)) == 0:
            shutil.rmtree(series_dir)
    if not os.path.isdir(series_dir):
        client = TCIAClient()
        # first download and extract data, then move to final destination
        # mv should be atomic, so existence of the final destination dir flags a successful download
        try:
            client.get_image(
                seriesInstanceUid=series["SeriesInstanceUID"],
                downloadPath=series_partial_dir,
                zipFileName="data.zip",
            )
            zipfile = os.path.join(series_partial_dir, "data.zip")
            subprocess.check_call(["unzip", zipfile, "-d", series_partial_dir])
            os.unlink(zipfile)
        except Exception as e:
            print(f"Error download: {e}")
            print(series)
            return

    if os.path.isdir(series_partial_dir):
        shutil.move(series_partial_dir, series_dir)

    if save_tags is not None:
        # only save tags per-series, not per-dicom, so we only need to check first file
        dicoms = glob.glob(os.path.join(series_dir, "*dcm"))
        if len(dicoms) > 0:
            f = pydicom.dcmread(dicoms[0])
            for tag, label in save_tags:
                if callable(tag):
                    series[label] = tag(f)
                else:
                    series[label] = getattr(f, tag)

    series["dicom_path"] = series_dir

    return series


def download_collection(collection_name, modality, download_path, save_tags=None):
    os.makedirs(download_path, exist_ok=True)
    client = TCIAClient()
    series = client.get_series(collection=collection_name, modality=modality)
    for s in series:
        s["PatientID"] = s["PatientID"].replace("ISPY2-", "")
    rows = p_umap(
        download_series,
        series,
        [download_path for _ in series],
        [save_tags for _ in series],
        num_cpus=args.cpus,
    )
    series_metadata = pd.DataFrame(rows)
    series_metadata.to_csv(
        os.path.join(download_path, "per_series_metadata.csv"), index=False
    )


def dicom_series_to_nifti(series, dicom_root, nifti_root):
    try:
        if "VOLSER" in series["SeriesDescription"]:
            dicom_series = os.path.join(
                dicom_root,
                series["PatientID"],
                series["StudyInstanceUID"],
                series["SeriesInstanceUID"],
            )
            os.makedirs(
                os.path.join(
                    nifti_root, series["PatientID"], series["StudyInstanceUID"]
                ),
                exist_ok=True,
            )
            nifti_partial_path = os.path.join(
                nifti_root,
                series["PatientID"],
                series["StudyInstanceUID"],
                series["SeriesInstanceUID"] + ".partial.nii.gz",
            )
            nifti_final_path = nifti_partial_path.replace("partial.nii.gz", "nii.gz")
            if os.path.isfile(nifti_partial_path):
                os.unlink(nifti_partial_path)
            if not os.path.isfile(nifti_final_path):
                dicom2nifti.dicom_series_to_nifti(
                    dicom_series,
                    nifti_partial_path,
                    reorient_nifti=True,
                )
                os.rename(nifti_partial_path, nifti_final_path)
            series["nifti_path"] = nifti_final_path
    except Exception as e:
        print(f"problem converting series {series}: {e}")
        series["error"] = e

    return series


def dicom_collection_to_nifti(dicom_path, nifti_path):
    os.makedirs(nifti_path, exist_ok=True)
    series = pd.read_csv(os.path.join(dicom_path, "per_series_metadata.csv"))
    series.PatientID = series.PatientID.astype(str)
    series.StudyInstanceUID = series.StudyInstanceUID.astype(str)
    series.SeriesInstanceUID = series.SeriesInstanceUID.astype(str)
    rows = p_umap(
        dicom_series_to_nifti,
        series.to_dict("records"),
        [dicom_path for _ in series.to_dict("records")],
        [nifti_path for _ in series.to_dict("records")],
        num_cpus=args.cpus,
    )
    series_metadata = pd.DataFrame(rows)

    series_metadata = pd.read_csv(os.path.join(nifti_path, "per_series_metadata.csv"))
    series_metadata["nifti_exists"] = series_metadata.nifti_path.apply(
        lambda x: os.path.isfile(str(x))
    )
    series_metadata.to_csv(
        os.path.join(nifti_path, "per_series_metadata.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download TCIA collection")
    parser.add_argument("collection", type=str)
    parser.add_argument("modality", type=str)
    parser.add_argument("dicom_path", type=str)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--nifti_path", type=str, default=None)
    parser.add_argument("--save_tags", type=str, nargs="+", default=None)

    args = parser.parse_args()

    # start = time.time()
    # download_collection(
    #     args.collection,
    #     args.modality,
    #     args.dicom_path,
    #     save_tags=[x.split(",") for x in args.save_tags]
    #     if args.save_tags is not None
    #     else None,
    # )
    # print(
    #     f"Done downloading {args.collection} in {((time.time() - start) / 60.):.0f} minutes."
    # )
    if args.nifti_path is not None:
        start = time.time()
        dicom_collection_to_nifti(args.dicom_path, args.nifti_path)
        print(
            f"Done converting dicom to nifti for {args.collection} in {((time.time() - start) / 60.):.0f} minutes."
        )
