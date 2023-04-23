import sys
from pathlib import Path
from train_mlm import to_str
import pandas as pd
from tqdm import tqdm

df = pd.read_parquet(Path(sys.argv[1]) / "unlabelled.snappy.parquet")

f = open("mlm.txt", "w")
for row_idx, row in tqdm(df.iterrows()):
    row_str = to_str(row)
    f.write(row_str)
    f.write("\n")

f.close()