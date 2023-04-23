from pathlib import Path
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from train import to_str


class BertDatasetTest(Dataset):

    def __init__(self, df, tokenizer, use_ua):
        self.df = df
        self.tokenizer = tokenizer
        self.use_ua = use_ua
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        return row.id, to_str(row)
    
    def collate_fn(self, batch):
        ids = [b[0] for b in batch]
        strings = [b[1] for b in batch]
        inp = self.tokenizer(strings, return_tensors="pt", max_length=512, padding=True, truncation=True)
        inp["_ids"] = ids
        return inp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1]).to(device)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
test_df = pd.read_parquet(Path(sys.argv[2]) / "test.parquet")


dataset = BertDatasetTest(test_df, tokenizer, False)
dataloader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)

ids = []
is_bot = []
for batch in tqdm(dataloader):
    ids += batch.pop("_ids")
    for k, v in batch.items():
        batch[k] = v.cuda()
    with torch.no_grad():
        preds = model(**batch)
        preds = torch.softmax(preds.logits, 1)[:, 1]
        preds = preds.tolist()

    is_bot += preds



pd.DataFrame({"id": ids, "is_bot": is_bot}).to_csv(sys.argv[3], index=None)
