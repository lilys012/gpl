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
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets.dataset_dict import DatasetDict
from datasets import Dataset

if __name__=='__main__':
    model_params = {
    "MODEL": "BeIR/query-gen-msmarco-t5-base-v1",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size (256) : 2epoch
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 4,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 200,  # max length of source tex (350)
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text (64)
    "SEED": 1001,  # set seed for reproducibility
    }
    console = Console(record=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--data_name', required=True)
    args = parser.parse_args()

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

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
    dataframe = pd.DataFrame({'corpus': t_corpus, 'query': t_query})

    source_text="corpus"
    target_text="query"

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    #display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.9
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    train_dataset = Dataset.from_dict(train_dataset)
    val_dataset = Dataset.from_dict(val_dataset)
    my_dataset_dict = DatasetDict({"train":train_dataset,"validation":val_dataset})

    # console.print(f"FULL Dataset: {dataframe.shape}")
    # console.print(f"TRAIN Dataset: {train_dataset.shape}")
    # console.print(f"TEST Dataset: {val_dataset.shape}\n")

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # def tokenize_function(examples, text, max_length):
    #     return tokenizer(examples, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    # tokenized_train_dataset_source = tokenize_function(list(train_dataset[source_text]), source_text, model_params["MAX_SOURCE_TEXT_LENGTH"])
    # tokenized_train_dataset_target = tokenize_function(list(train_dataset[target_text]), target_text, model_params["MAX_TARGET_TEXT_LENGTH"])
    # tokenized_val_dataset_source = tokenize_function(list(val_dataset[source_text]), source_text, model_params["MAX_SOURCE_TEXT_LENGTH"])
    # tokenized_val_dataset_target = tokenize_function(list(val_dataset[target_text]), target_text, model_params["MAX_TARGET_TEXT_LENGTH"])
    # tokenized_datasets = {"train" : [tokenized_train_dataset_source, tokenized_train_dataset_target], "validation" : [tokenized_val_dataset_source, tokenized_train_dataset_target]}
    
    # console.print(f"{tokenized_datasets['train'][:2]}")

    # raw_datasets = load_dataset("glue", "mrpc")
    # checkpoint = "bert-base-uncased"
    # atokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # def atokenize_function(example):
    #     return atokenizer(example["sentence1"], example["sentence2"], truncation=True)
    # atokenized_datasets = raw_datasets.map(atokenize_function, batched=True)
    # console.print(f"{raw_datasets['train'][0]}")
    # console.print(f"{atokenized_datasets['train'][0]}")

    def preprocess_function(examples):
        inputs = examples[source_text]
        targets = examples[target_text]
        model_inputs = tokenizer(inputs, max_length=model_params["MAX_SOURCE_TEXT_LENGTH"], truncation=True, padding="max_length")
        labels = tokenizer(targets, text_target=targets, max_length=model_params["MAX_TARGET_TEXT_LENGTH"], padding='max_length', truncation=True)

        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = my_dataset_dict["train"].map(preprocess_function, batched=True, remove_columns=my_dataset_dict["train"].column_names)
    #tokenized_datasets = preprocess_function(list(train_dataset[source_text]), list(train_dataset[target_text]), max_length=model_params["MAX_SOURCE_TEXT_LENGTH"], max_target_length=model_params["MAX_TARGET_TEXT_LENGTH"])
    console.print(f"{tokenized_datasets[0]}")