from pyarrow.dataset import dataset, FileSystemDataset
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch
import numpy as np

from pathlib import Path
import os
import string
import json
import re

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.vocab import *

DATASET_ROOT = "/mnt/sda2/dataset/wikitext-raw/"
DATASET_OUTPUT = "/mnt/sda2/dataset/wikitext-tokenized/"

def _get_parquest_list(split=None):
    file_paths = []
    for file in os.listdir(DATASET_ROOT):
        if file.endswith(".parquet"):
            if split:
                if file.startswith(split):
                    file_paths.append(file)
            else:
                file_paths.append(file)
    return file_paths

class ParquetDataset(IterableDataset):
    def __init__(self, file_paths, batch_size):
        self.file_paths: list = file_paths
        self.raw_dataset = dataset(file_paths, format="parquet", filesystem=DATASET_ROOT)
        self.batch_size = batch_size

    def __iter__(self):
        for record_batch in self.raw_dataset.to_batches(
            batch_size=self.batch_size
            ):
            yield list(map(str, record_batch["text"]))


def normalize(batch):
    # print(batch)
    # assert False
    data = []
    for sample in batch:
        temp = ""
        sentence = []
        for ch in sample:
            if ch != " ":
                temp += ch
            else:
                if temp:
                    sentence.append(temp.lower())
                    temp = ""
                sentence.append(" ")
        if temp:
            sentence.append(temp.lower())
        
        data.append(sentence)
    return data

def pre_tokenize(batch):
    data = []
    for sample in batch:
        sentence = []
        for word in sample:
            new_word = []
            # Split using every punctuation
            temp = ""
            for ch in word:
                if ch not in string.punctuation:
                    temp += ch
                else:
                    if temp:
                        new_word.append(temp)
                        temp = ""
                    new_word.append(ch)
            if temp:
                new_word.append(temp)
            
            # Append to the result sentence
            for part in new_word:
                sentence.append(part)
        data.append(sentence)
    return data

def post_process(batch, vocab_enc):
    data = []
    for sample in batch:
        sentence = []
        for word in sample:
            sentence.append(vocab_enc.get(word, vocab_enc[UNK]))
        data.append(sentence)
    return data

def calculate_vocab(dataset, sample_batch=100, maximum_vocab=10000):
    """Return a list of vocabuary.
    If sample_batch <= 0, we will use the entire dataset to calculate vocabulary.
    """
    freq = {}
    if sample_batch > 0:
        for _ in range(sample_batch):
            batch = next(iter(dataset))
            data = pre_tokenize(normalize(batch))
            for sentence in data:
                for word in sentence:
                    freq[word] = freq.get(word, 0) + 1
    else: 
        for batch in dataset:
            data = pre_tokenize(normalize(batch))
            for sentence in data:
                for word in sentence:
                    freq[word] = freq.get(word, 0) + 1
    freq_list = sorted(freq.items(), key=lambda item: item[1], reverse=True) # sort by value
    vocab = SPECIAL_TOKS
    for k,v in freq_list[:maximum_vocab-len(SPECIAL_TOKS)]:
        vocab.append(k)
    
    return vocab

    
if __name__ == "__main__":
    # Vocab mapping dicts
    vocab_enc = None
    vocab_dec = None

    file_paths = _get_parquest_list()
    test_dataset = ParquetDataset(file_paths, batch_size=32)

    if Path("vocab_dec.json").exists() or Path("vocab_enc.json").exists():
        print("Vocabuary file already exists, are you sure to override?")
        i = input()
        if i.strip().lower() != "y" and i.strip().lower() != "yes":
            with open("vocab_dec.json", "r") as f:
                vocab_dec = json.load(f)
            with open("vocab_enc.json", "r") as f:
                vocab_enc = json.load(f)
    
    if not vocab_enc:
        print("Generating vocabulary...")
        vocab = calculate_vocab(test_dataset, sample_batch=-1, maximum_vocab=16384)
        vocab_dec = {
            id:word for id, word in enumerate(vocab)
        }
        vocab_enc = {
            word:id for id, word in enumerate(vocab)
        }

        with open(Path(__file__).parent / "vocab_dec.json", "w") as f:
            json.dump(vocab_dec, f)
        with open(Path(__file__).parent / "vocab_enc.json", "w") as f:
            json.dump(vocab_enc, f)

    if not Path(DATASET_OUTPUT).exists():
        os.mkdir(DATASET_OUTPUT)
    for split in ["train", "test", "valid"]:
        print(f"Processing split {split}...")
        file_paths = _get_parquest_list(split)
        sub_dataset = ParquetDataset(file_paths, 32)
        processed_data = []
        for batch in sub_dataset:
            data = post_process(pre_tokenize(normalize(batch)), vocab_enc)

            for sentence, raw_sentence in zip(data, batch):
                for token in sentence:
                    processed_data.append(token)
                # Check for titles
                if re.match(r"/^=\s*(.+?)\s*=$/gm", raw_sentence):
                    print(f"Found a heading: {raw_sentence}")
                    # Insert a EOS
                    processed_data.append(vocab_enc[EOS])

        # Write to binary
        processed_data = np.array(processed_data, dtype=np.uint16)
        processed_data.dump(Path(DATASET_OUTPUT) / f"{split}.bin")

    