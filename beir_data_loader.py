from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
import random

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_1000(self, split="test", num=1000) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            random.seed(1001)
            list_all = []
            for qid in self.qrels.keys():
                for cid in self.qrels[qid].keys():
                    if self.qrels[qid][cid] != 0:
                        list_all.append({qid: cid})
            list_all = random.sample(list_all, num if num < len(self.qrels) else len(self.qrels))
            new_qrels = {}
            for item in list_all:
                for qid, cid in item.items():
                    score = self.qrels[qid][cid]
                    if qid not in new_qrels:
                        new_qrels[qid] = {cid: score}
                    else:
                        new_qrels[qid][cid] = score
            self.queries = {qid: self.queries[qid] for qid in new_qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        return self.corpus, self.queries, new_qrels

    def _load_corpus(self):

        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = line.get("title") + " " + line.get("text") 

    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):

        reader = csv.reader(open(self.qrels_file, encoding="utf-8"),
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
