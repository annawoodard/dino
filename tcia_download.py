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
from retry import retry
from p_tqdm import p_umap


@retry(tries=3, delay=1, backoff=2)
def download_series(series, download_path, client, save_tags):
    series_dir = os.path.join(
        download_path,
        series["PatientID"],
        series["StudyInstanceUID"],
        series["SeriesInstanceUID"],
    )
    series_partial_dir = series_dir + ".partial"
    if os.path.isdir(series_partial_dir):
        os.unlink(series_partial_dir)  # cleanup previous download attempt
    if not os.path.isdir(series_dir):
        # first download and extract data, then move to final destination
        # mv should be atomic; so the final destination dir flags a successful download
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

    if save_tags is not None:
        rows = []
        for dicom in glob.glob(os.path.join(series_partial_dir, "*dcm")):
            row = series.copy()
            row["path"] = dicom
            if save_tags is not None:
                f = pydicom.dcmread(dicom)
                for tag, label in save_tags:
                    if callable(tag):
                        row[label] = tag(f)
                    else:
                        row[label] = getattr(f, tag)
            rows.append(row)

    if os.path.isdir(series_partial_dir):
        shutil.move(series_partial_dir, series_dir)

    return None if save_tags is None else rows


def download_collection(collection_name, modality, download_path, save_tags=None):
    os.makedirs(download_path, exist_ok=True)
    client = TCIAClient()
    series = client.get_series(collection=collection_name, modality=modality)
    series_metadata = pd.DataFrame(series)
    series_metadata.to_csv(
        os.path.join(download_path, "per_series_metadata.csv"), index=False
    )
    row_chunks = p_umap(
        download_series,
        series,
        [download_path for _ in series],
        [client for _ in series],
        [save_tags for _ in series],
        num_cpus=args.cpus,
    )
    if save_tags is not None:
        rows = sum(row_chunks, [])
        metadata = pd.DataFrame(rows)
        metadata.to_csv(
            os.path.join(download_path, "per_dicom_metadata.csv"), index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download TCIA collection")
    parser.add_argument("collection", type=str)
    parser.add_argument("modality", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("--cpus", type=int, default=1)

    args = parser.parse_args()

    start = time.time()
    download_collection(args.collection, args.modality, args.path)
    print(
        f"Done downloading {args.collection} in {((time.time() - start) / 60.):.0f} minutes."
    )
