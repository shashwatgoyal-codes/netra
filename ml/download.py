"""Download and extract APTOS 2019 into data/raw.

Needs ~/.kaggle/kaggle.json and the competition rules accepted on kaggle.com
(https://www.kaggle.com/c/aptos2019-blindness-detection/rules).

Usage: uv run python download.py
"""

import zipfile

from common import RAW

COMPETITION = "aptos2019-blindness-detection"


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # authenticates on import

    RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for name in ("train.csv", "train_images"):
        if (RAW / name).exists():
            continue
        archive = RAW / f"{COMPETITION}.zip"
        if not archive.exists():
            api.competition_download_files(COMPETITION, path=str(RAW), quiet=False)
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if m == "train.csv" or m.startswith("train_images/")]
            z.extractall(RAW, members)
        archive.unlink()
        break
    print(f"train images: {len(list((RAW / 'train_images').glob('*.png')))}")


if __name__ == "__main__":
    main()
