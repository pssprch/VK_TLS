from pathlib import Path
from argparse import ArgumentParser
import torch
import deepspeed
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import enable_full_determinism


def to_str(row):
    return " ".join([row.ua] + eval(row.ciphers.decode()) + eval(row.curves.decode()))
    

class BertDataset(Dataset):

    def __init__(self, df, tokenizer, use_ua):
        self.df = df
        self.tokenizer = tokenizer
        self.use_ua = use_ua
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ind):
        row = self.df.iloc[ind]
        if "label" in row:
            label = row.label
        else:
            label = None
        return to_str(row), label
    
    def collate_fn(self, batch):
        strings = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        inp = self.tokenizer(strings, return_tensors="pt", max_length=512, padding=True, truncation=True)
        if labels[0] is not None:
            labels = torch.LongTensor(labels)
            inp["labels"] = labels
        return inp

def roc_auc(inp):
    preds = inp.predictions
    labels = inp.label_ids
    with torch.no_grad():
        preds = torch.from_numpy(preds)
        preds = torch.softmax(preds, 1)[:, 1]
        classes = (preds > 0.5).long().numpy()
        preds = preds.numpy()
    

    d = {"auc": roc_auc_score(labels, preds)}
    d["accuracy"] = (classes == labels).sum() / len(classes)
    d["accuracy_0"] = ((classes == labels) & (labels == 0)).sum() / ((labels == 0).sum() + 1e-9)
    d["accuracy_1"] = ((classes == labels) & (labels == 1)).sum() / ((labels == 1).sum() + 1e-9)
    return d

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_df = pd.read_parquet(args.data / "train_split.parquet")
    val_df = pd.read_parquet(args.data / "val_split.parquet")
    train_dataset = BertDataset(train_df, tokenizer, args.use_ua)
    val_dataset = BertDataset(val_df, tokenizer, args.use_ua)
    tr_args = TrainingArguments(
        output_dir=args.name,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        local_rank=args.local_rank,
        eval_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=20,
        warmup_ratio=0,
        logging_dir=Path(args.name) / "logs",
        lr_scheduler_type="constant",
        seed=1234,
        data_seed=1234,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        greater_is_better=True
    )
    trainer = Trainer(
        model,
        tr_args,
        train_dataset.collate_fn,
        train_dataset,
        val_dataset,
        tokenizer,
        compute_metrics=roc_auc,
    )
    trainer.train()


if __name__ == "__main__":
    deepspeed.init_distributed(dist_backend="nccl")
    enable_full_determinism(0)
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--use-ua", action="store_true")
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    main(args)

