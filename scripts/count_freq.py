"""检查数据集中各个label的频率"""

from typing import Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.preprocess import DATASET_OUTPUT
from dataset.vocab import *
from model.banana import BananaConfig

class BananaLoader(IterableDataset):
    def __init__(self, split:str, config:BananaConfig) -> None:
        super().__init__()
        self.config = config
        self.path = Path(DATASET_OUTPUT) / f"{split}.bin"
        self.batch_size = config.batch_size

        if not self.path.exists():
            raise AttributeError(f"Split {split} not found!") 
        
        self.data = np.load(self.path, allow_pickle=True)
        self.length = len(self.data)
    
    def __iter__(self) -> Iterator:
        while True:
            indices = np.random.choice(self.length, self.batch_size)
            X = np.ndarray((self.batch_size, self.config.seq_len), dtype=int)
            Y = np.ndarray((self.batch_size, self.config.seq_len), dtype=int)
            mask_inv = np.zeros((self.batch_size, self.config.seq_len), dtype=int)

            for i in range(self.batch_size):
                if indices[i] == self.length - 1:
                    indices[i] = 0 # we cannot handle Y
            
                # Desired range [st, ed), we first extract seq_len + 1 elements
                st = indices[i]
                ed = min(st + self.config.seq_len + 1, self.length)
                
                eos_idx = np.flatnonzero(self.data[st:ed] == eos) # indices of non-zero element
                if eos_idx.size > 0:
                    ed = st + int(eos_idx[0]) + 1
                
                # X: [st, ed_x]
                ed_x = ed - 1
                actual_len = ed_x - st
                X[i, :] = np.pad(self.data[st:ed_x], (0, self.config.seq_len - actual_len), constant_values=pad)

                # Y: [st_y, ed]
                st_y = st + 1
                actual_len = ed - st_y
                Y[i, :] = np.pad(self.data[st_y:ed], (0, self.config.seq_len - actual_len), constant_values=pad)

                mask_inv[0 : ed_x - st] = 1 # valid numbers
            
            mask = (mask_inv == 0) # inverse valid mask
            Y_masked = np.where(mask, -100, Y) # pad should not generate loss

            yield torch.tensor(X), torch.tensor(Y_masked)

if __name__ == "__main__":
    config = BananaConfig()
    config.batch_size = 6
    config.seq_len = 100
    train_loader = BananaLoader("train", config)
    enc, dec = load_vocab()

    freq_counter = {}
    cnt = 0
    for batch in train_loader:
        cnt += 1
        if cnt > 100:
            break
        print(f"batch {cnt}")
        X, Y = batch
        for i in range(config.batch_size):
            raw_x = []
            sentence_x = []
            raw_y = []
            sentence_y = []
            for j in range(config.seq_len):
                x = str(int(X[i, j]))
                y = str(int(Y[i, j]))
                raw_x.append(x)
                raw_y.append(y)
                sentence_x.append(dec[x])
                sentence_y.append(dec[y])

                freq_counter[x] = freq_counter.get(x, 0) + 1
            # print("Batch ", i)
            # print(repr("".join(sentence_x)))
            # print(repr("".join(sentence_y)))
            # print(" ".join(raw_x))
            # print(" ".join(raw_y))
            # print(" ".join(mask[i, :]))
            # print()
            # input()

    freq_sorted = sorted([(v, k) for k, v in freq_counter.items()], reverse=True)
    freq_sorted = freq_sorted[:50]
    totals = 0
    for v, k in freq_sorted:
        totals += v
    for v, k in freq_sorted:
        print(f"{k}({dec[k]}: {v/totals : .3f})")

