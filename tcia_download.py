import os
import glob
import pydicom
from tciaclient.core import TCIAClient
import pandas as pd
import subprocess
from tqdm import tqdm


def download_collection(collection_name, modality, download_path, save_tags=None):
    client = TCIAClient()
    series = client.get_series(collection=collection_name, modality=modality)
    metadata = []
    for s in tqdm(series):
        series_dir = os.path.join(
            download_path, s["PatientID"], s["StudyInstanceUID"], s["SeriesInstanceUID"]
        )
        client.get_image(
            seriesInstanceUid=s["SeriesInstanceUID"],
            downloadPath=series_dir,
            zipFileName="data.zip",
        )
        zipfile = os.path.join(series_dir, "data.zip")
        subprocess.check_call(["unzip", zipfile, "-d", series_dir, ">&", "/dev/null"])
        os.unlink(zipfile)
        for dicom in glob.glob(os.path.join(series_dir, "*dcm")):
            row = s.copy()
            row["path"] = dicom
            if save_tags is not None:
                f = pydicom.dcmread(dicom)
                for tag, label in save_tags:
                    if callable(tag):
                        row[label] = tag(f)
                    else:
                        row[label] = getattr(f, tag)

            metadata.append(row)
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(os.path.join(download_path, "metadata.csv"), index=False)
