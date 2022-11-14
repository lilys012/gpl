import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from rich.table import Column, Table
from rich import box
from rich.console import Console
from ftQG.MyDataClass import MyDataClass
from ftQG.train import train
from ftQG.validate import validate
from beir_data_loader import GenericDataLoader
import argparse
from torch import cuda
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
# import evaluate

def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

def valTrainer(dataframe, source_text, target_text, model_params, output_dir="./earlyStopT5/outputs/"):
    device = 'cuda' if cuda.is_available() else 'cpu'
    console = Console(record=True)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.9
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    train_dataset = Dataset.from_dict(train_dataset)
    val_dataset = Dataset.from_dict(val_dataset)
    my_dataset_dict = DatasetDict({"train":train_dataset,"validation":val_dataset})

    def preprocess_function(examples):
        inputs = examples[source_text]
        targets = examples[target_text]
        model_inputs = tokenizer(inputs, max_length=model_params["MAX_SOURCE_TEXT_LENGTH"], truncation=True, padding="max_length")
        labels = tokenizer(targets, text_target=targets, max_length=model_params["MAX_TARGET_TEXT_LENGTH"], padding='max_length', truncation=True)

        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = my_dataset_dict["train"].map(preprocess_function, batched=True, remove_columns=my_dataset_dict["train"].column_names)
    tokenized_val_dataset = my_dataset_dict["validation"].map(preprocess_function, batched=True, remove_columns=my_dataset_dict["validation"].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # metric = evaluate.load("sacrebleu")

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     # In case the model returns more than the prediction logits
    #     if isinstance(preds, tuple):
    #         preds = preds[0]

    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    #     # Replace -100s in the labels as we can't decode them
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     # Some simple post-processing
    #     decoded_preds = [pred.strip() for pred in decoded_preds]
    #     decoded_labels = [[label.strip()] for label in decoded_labels]

    #     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     return {"bleu": result["score"]}

    args = Seq2SeqTrainingArguments(
        "saved_model",
        evaluation_strategy = "steps",
        eval_steps = 5,
        load_best_model_at_end = True,
        learning_rate=2e-5,
        per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=model_params["VALID_BATCH_SIZE"],
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=128,
        predict_with_generate=True,
        fp16=False,
        #report_to='wandb',
        #run_name="ut_del_three_per_each_ver2_early_stop_4"  # name of the W&B run (optional)
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

if __name__=='__main__':
    model_params = {
    "MODEL": "BeIR/query-gen-msmarco-t5-base-v1",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size (256) : 2epoch
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 4,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 350,  # max length of source tex (350)
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text (64)
    "SEED": 1001,  # set seed for reproducibility
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--data_name', required=True)
    args = parser.parse_args()

    corpus, all_queries, train_qrels = GenericDataLoader(args.data_path).load_1000(split='train', num=1000)
    t_corpus = []
    t_query = []
    gen_prefix=""
    for qid in train_qrels.keys():
        for cid in train_qrels[qid].keys():
            if(train_qrels[qid][cid] != 0):
                t_corpus.append(gen_prefix+corpus[cid])
                t_query.append(all_queries[qid])
    print("{} {} training sets".format(len(t_corpus), len(t_query)))
    df = pd.DataFrame({'corpus': t_corpus, 'query': t_query})

    valTrainer(
        dataframe=df,
        source_text="corpus",
        target_text="query",
        model_params=model_params,
        output_dir="T5_outputs/{}".format(args.data_name),
    )
