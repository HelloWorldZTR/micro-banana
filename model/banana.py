import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np

class BananaConfig:
    # Model
    seq_len: int = 256 # N
    vocab_size: int = 32_768 # vocab size
    transformer_blocks: int = 6
    heads: int = 1 # K
    latent_dim: int = 128 # Hidden size D for transformer blocks
    hidden_dim: int = 512 # Bottle neck size for FFNN
    drop_out: float = 0.3 # Dropout rate for FFNN

    # Data
    batch_size: int = 16

    # Trainging
    lr: float = 1e-4
    device: str = "cuda:0"
    epoches: int = 1
    log_every: int = 100 # steps
    eval_every: int = 1 # steps
    valid_every: int = 1 # epoches
    ckpt_every: int = 10000 # steps


class TransformerBlock(Module):
    def __init__(self, config: BananaConfig):
        super().__init__()
        self.config = config
        # LN
        self.ln1 = nn.LayerNorm(config.latent_dim)
        self.ln2 = nn.LayerNorm(config.latent_dim)

        # Self-Attention
        self.w_Q = nn.Linear(config.latent_dim, config.latent_dim)
        self.w_K = nn.Linear(config.latent_dim, config.latent_dim)
        self.w_V = nn.Linear(config.latent_dim, config.latent_dim)
        self.softmax = nn.Softmax(-1)

        mask = np.tril(np.ones((config.seq_len, config.seq_len), dtype=bool))
        mask = np.where(mask, np.zeros((config.seq_len, config.seq_len)), -np.inf)
        mask = torch.tensor(mask, requires_grad=False, dtype=torch.float32)
        self.register_buffer("attn_mask", mask)
        
        # FFNN
        self.FFNN = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.drop_out),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )

    def forward(self, z):
        # Self-Attention /w skip connection
        z_skip_1 = z
        z = self.ln1(z)

        Q = self.w_Q(z)
        K = self.w_K(z)
        V = self.w_V(z)
        d_k = self.config.heads

        masked = Q @ K.permute(0, 2, 1) + self.attn_mask
        attn_score = self.softmax(masked) / np.sqrt(d_k)
        z = attn_score @ V + z_skip_1

        # FFNN /w skip connection
        z_skip_2 = z
        z = self.ln2(z)

        z = self.FFNN(z) + z_skip_2

        return z

class Banana(Module):
    """
    Input: a sequence of token ids
    Output: predicted token id logits
    """
    def __init__(self, config: BananaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.latent_dim)

        emb = []
        for i in range(config.seq_len):
            t = torch.arange(0, config.latent_dim, 1)
            even = (t % 2 == 0)
            odd = (t % 2 == 1)
            row = torch.sin(i / torch.pow(10000, 2*t / config.latent_dim)) * even + \
                torch.cos(i / torch.pow(10000, 2*t / config.latent_dim)) * odd
            emb.append(row)
        self.register_buffer("pe", torch.vstack(emb))

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) 
                                   for _ in range(config.transformer_blocks)])

        self.output = nn.Linear(config.latent_dim, config.vocab_size)

    def forward(self, inputs):
        # Embedding
        # BxNx1 => BxNxD
        x = self.embedding(inputs)

        # Positional embedding
        x = x + self.pe

        # Transformer blocks 
        # BxNxD => BxNxD
        for block in self.transformer_blocks:
            x = block(x)

        # Output logits
        # BxNxD => BxNx(Vocab Size)
        logits = self.output(x)

        return logits


if __name__ == "__main__":
    config = BananaConfig()
    model = Banana(config)
