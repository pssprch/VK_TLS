from pathlib import Path
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

def to_str(row):
    return " ".join([row.ua] + row.ciphers.tolist() + row.curves.tolist())

class BertDatasetTest(Dataset):

    def __init__(self, df, tokenizer, use_ua):
        self.df = df
        self.tokenizer = tokenizer
        self.use_ua = use_ua
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        return to_str(row)
    
    def collate_fn(self, batch):
        strings = batch
        inp = self.tokenizer(strings, return_tensors="pt", max_length=512, padding=True, truncation=True)
        return inp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1]).to(device)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
test_df = pd.read_parquet(Path(sys.argv[2]) / "unlabelled.snappy.parquet")


dataset = BertDatasetTest(test_df, tokenizer, False)
dataloader = DataLoader(dataset, 16, collate_fn=dataset.collate_fn)

is_bot = []
for idx, batch in tqdm(enumerate(dataloader)):
    for k, v in batch.items():
        batch[k] = v.cuda()
    with torch.no_grad():
        preds = model(**batch)
        preds = torch.softmax(preds.logits, 1)[:, 1]
        preds = preds.tolist()

    is_bot += preds
    if idx % 1000 == 0:
        print("dumping", idx)
        markup = test_df.copy()
        markup = markup.iloc[:len(is_bot)]
        markup["label"] = is_bot
        markup.to_parquet("markedup.parquet", index=None)


