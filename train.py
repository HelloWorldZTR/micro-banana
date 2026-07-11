import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import pickle
import os
import time
from pathlib import Path

from model import Banana, BananaConfig
from dataloader import BananaLoader

config = BananaConfig()
device = torch.device(config.device)

# Dataloader
train_loader = BananaLoader("train", config)
test_loader = BananaLoader("test", config)
valid_loader = BananaLoader("valid", config)

# Model and optimizer
model = Banana(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
loss_fn = F.cross_entropy

# Shorthand notation
B = config.batch_size
N = config.seq_len

OUTPUT_DIR = Path(__file__).parent / "output" / time.strftime("%Y-%m-%d %H:%m")
if not OUTPUT_DIR.exists():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    step = 0
    for epoch in range(config.epoches):
        print(f"Epoch {epoch}/{config.epoches}")

        model.train()
        # for batch in train_loader:
        batch = next(iter(train_loader))
        for i in range(1000000):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits.reshape(B*N, -1), Y.reshape(B*N))
            loss.backward()
            optimizer.step()

            if step % config.log_every == 0:
                print(f"Step: {step}, loss={loss.detach().cpu():.4f}")
            if (step + 1) % config.eval_every == 0:
                pass
            if (step + 1) % config.ckpt_every == 0:
                print(f"Saved checkpoint to {OUTPUT_DIR}")
                with open(OUTPUT_DIR / f"step_{step + 1}.pkl", "wb") as f:
                    pickle.dump(model, f)
            step += 1

        if epoch + 1 % config.valid_every == 0:
            pass





