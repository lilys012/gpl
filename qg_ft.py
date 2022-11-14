from beir_data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
import os
import argparse
import torch, logging, math, queue
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict
from tensorflow.keras.optimizers import Adam

def finetune(data_path, model_path='BeIR/query-gen-msmarco-t5-base-v1', gen_prefix: str = "", use_fast: bool = True):

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use pytorch device: {}".format(device))

    corpus, all_queries, train_qrels = GenericDataLoader(data_path).load_1000(split='train', num=10) 

    tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=use_fast)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    t_corpus = []
    t_query = []
    for qid in train_qrels.keys():
        for cid in train_qrels[qid].keys():
            if(train_qrels[qid][cid] != 0):
                t_corpus.append(gen_prefix+corpus[cid])
                t_query.append(all_queries[qid])
    print("{} {} training sets".format(len(t_corpus), len(t_query)))

    encoding = tokenizer(t_corpus, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    labels = tokenizer(t_query, padding=True, truncation=True, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    model.fit(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
    finetune(args.data_path)
