import json
from pathlib import Path

# Special token strings
UNK = "<UNK>"
CLS = "<CLS>"
SEP = "<SEP>"
PAD = "<PAD>"
MASK = "<MASK>"
EOS = "<EOS>"

SPECIAL_TOKS = [UNK, CLS, SEP, PAD, MASK, EOS]

def load_vocab():
    """Returns encode dict, decode dict

    ```python
    enc, dec = load_vocab()
    token_id = enc[token]
    token = dec[token_id]
    ```
    """
    with open(Path(__file__).parent / "vocab_enc.json", "r") as f:
        enc = json.load(f)
    with open(Path(__file__).parent / "vocab_dec.json", "r") as f:
        dec = json.load(f)
    return enc, dec

enc, dec = load_vocab()

# Special token ids
unk = enc[UNK]
cls = enc[CLS]
sep = enc[SEP]
pad = enc[PAD]
mask = enc[MASK]
eos = enc[EOS]