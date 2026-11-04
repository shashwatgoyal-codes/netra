"""Crop and cache APTOS 2019 images, then make a stratified train/val/test split.

Expects data/raw/train.csv and data/raw/train_images/*.png (the Kaggle zip,
extracted). Writes data/processed/<id_code>.jpg and data/splits.csv.
"""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import CACHE_SIZE, PROCESSED, RAW, SPLITS, crop_retina

SEED = 42


def process(id_code: str) -> None:
    out = PROCESSED / f"{id_code}.jpg"
    if out.exists():
        return
    img = Image.open(RAW / "train_images" / f"{id_code}.png")
    crop_retina(img).resize((CACHE_SIZE, CACHE_SIZE), Image.LANCZOS).save(out, quality=95)


def main() -> None:
    df = pd.read_csv(RAW / "train.csv")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        list(tqdm(pool.map(process, df.id_code, chunksize=16), total=len(df), desc="crop"))

    # 80 / 10 / 10, stratified on grade so rare grades appear in every split.
    train, rest = train_test_split(df, test_size=0.2, stratify=df.diagnosis, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.diagnosis, random_state=SEED)
    splits = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    splits.to_csv(SPLITS, index=False)
    print(splits.groupby(["split", "diagnosis"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
