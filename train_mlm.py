import os
from pathlib import Path
from argparse import ArgumentParser
import torch
import deepspeed
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pandas as pd
from transformers import enable_full_determinism
from transformers import DataCollatorForLanguageModeling




def to_str(row):
    return " ".join([row.ua] + row.ciphers.tolist() + row.curves.tolist())

class LineByLineTextDataset(Dataset):
    # from transformers import LineByLineTextDataset

    def __init__(self, tokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.lines = lines
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i]
        t
        return {"input_ids": torch.LongTensor(self.tokenizer(line, add_special_tokens=True, truncation=True, max_length=self.block_size).input_ids)}

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_dataset = LineByLineTextDataset(tokenizer, args.data / "mlm_train.txt", 500)
    val_dataset = LineByLineTextDataset(tokenizer, args.data / "mlm_val.txt", 500)
    tr_args = TrainingArguments(
        output_dir=args.name,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        local_rank=args.local_rank,
        eval_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=500,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=2,
        warmup_ratio=0,
        logging_dir=Path(args.name) / "logs",
        lr_scheduler_type="constant",
        seed=1234,
        data_seed=1234,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model,
        tr_args,
        DataCollatorForLanguageModeling(tokenizer, mlm=True),
        train_dataset,
        val_dataset,
        tokenizer,
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

