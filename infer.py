import torch
import numpy as np
import pickle

from dataset.vocab import load_vocab
from dataset.preprocess import *

from model import Banana, BananaConfig
from dataloader import BananaLoader

config = BananaConfig()
device = torch.device(config.device)

B = config.batch_size
N = config.seq_len

model_path = Path(__file__).parent / "output" / "2026-07-11 17:07" / "step_30000.pkl"

def _decode(X):
    """Decode a model output X"""
    words = []
    for i in range(N):
        token_id = X[0, i]
        word = dec[str(int(token_id))]
        if word in (EOS, PAD):
            break
        words.append(word)
    return "".join(words)

def _format_tokens(X, valid_len):
    """Extract tokens from a model output X"""
    return list(map(int, X[0, :valid_len]))

def infer(model, tokens):
    # Pad to BxNx1
    X = np.full((B,N), pad)
    for i, token in enumerate(tokens[0]):
        X[0, i] = token
    X = torch.tensor(X)
    X = X.to(device)
    valid_length = len(tokens[0])

    eos_encountered = False
    while not eos_encountered:
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        y = torch.argmax(probs, dim=-1)
        y = y.detach().cpu()

        words = []
        out_ids = []
        in_ids = []
        for i in range(N):
            input_token = X[0, i]
            output_token = y[0, i]
            word = dec[str(int(output_token))]
            if word in (EOS, PAD):
                eos_encountered = (word == EOS)
                break
            # For debugging purpose
            in_ids.append(str(int(input_token)))
            out_ids.append(str(int(output_token)))
            words.append(word)
        
        X[0, valid_length] = y[0, valid_length]
        valid_length += 1
        if valid_length >= N:
            break
        print(_decode(X))
        print(_format_tokens(X, valid_length))
        input("Press Enter for next iteration")


if __name__ == "__main__":
    print(f"Model file is: {model_path}")
    sentence = input("Input your query:")
    sentence = [sentence]
    enc, dec = load_vocab()

    tokens = post_process(pre_tokenize(normalize(sentence)), enc)
    print("Tokens", tokens)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()

    infer(model, tokens)