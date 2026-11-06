"""Crop and cache training photos, then write the train/val/test split.

Sources:
  APTOS 2019  data/raw/train.csv + train_images/*.png  (Aravind, India)
              stratified 80/10/10 into train/val/test
  DDR         data/ddr_full/DDR-dataset/DR_grading/{train,valid}  (147 hospitals, China)
              its own train -> train, valid -> val; ungradable (grade 5) dropped.
              DDR's test split is never used here: external.py keeps it for evaluation.

Training on both teaches the model more than one camera and one grading
team; the first APTOS-only model caught 95% of referable eyes on APTOS and
37% on DDR.

Writes data/processed/<id>.jpg and data/splits.csv (id_code, diagnosis,
split, source, raw).
"""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from common import CACHE_SIZE, DATA, PROCESSED, RAW, SPLITS, crop_retina

SEED = 42
DDR = DATA / "ddr_full" / "DDR-dataset" / "DR_grading"


def process(job: tuple[str, str]) -> None:
    id_code, raw = job
    out = PROCESSED / f"{id_code}.jpg"
    if out.exists():
        return
    img = Image.open(DATA / raw)
    crop_retina(img).resize((CACHE_SIZE, CACHE_SIZE), Image.LANCZOS).save(out, quality=95)


def aptos() -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    df["source"] = "aptos"
    df["raw"] = [str((RAW / "train_images" / f"{i}.png").relative_to(DATA)) for i in df.id_code]
    # 80 / 10 / 10, stratified on grade so rare grades appear in every split.
    train, rest = train_test_split(df, test_size=0.2, stratify=df.diagnosis, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.diagnosis, random_state=SEED)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])


def ddr() -> pd.DataFrame:
    if not DDR.exists():
        return pd.DataFrame()
    parts = []
    for name, split in (("train", "train"), ("valid", "val")):
        df = pd.read_csv(DDR / f"{name}.txt", sep=" ", header=None, names=["file", "diagnosis"])
        df = df[df.diagnosis <= 4]
        parts.append(
            pd.DataFrame(
                {
                    "id_code": "ddr_" + df.file.str.rsplit(".", n=1).str[0],
                    "diagnosis": df.diagnosis,
                    "split": split,
                    "source": "ddr",
                    "raw": [str((DDR / name / f).relative_to(DATA)) for f in df.file],
                }
            )
        )
    return pd.concat(parts)


def main() -> None:
    splits = pd.concat([aptos(), ddr()], ignore_index=True)[["id_code", "diagnosis", "split", "source", "raw"]]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    jobs = list(zip(splits.id_code, splits.raw))
    with ProcessPoolExecutor() as pool:
        list(tqdm(pool.map(process, jobs, chunksize=16), total=len(jobs), desc="crop"))
    splits.to_csv(SPLITS, index=False)
    print(splits.groupby(["source", "split", "diagnosis"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
