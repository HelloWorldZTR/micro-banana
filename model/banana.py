import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np

class BananaConfig:
    # Model
    seq_len: int = 1024 # N
    vocab_size: int = 16_384 # vocab size
    transformer_blocks: int = 6
    heads: int = 6 # K
    latent_dim: int = 64 # Hidden size D for transformer blocks
    hidden_dim: int = 32 # Bottle neck size for FFNN
    drop_out: float = 0.3 # Dropout rate for FFNN

    # Data
    batch_size: int = 64

    # Trainging
    lr: float = 1e-4
    device = torch.device("mps")
    

class TransformerBlock(Module):
    def __init__(self, config: BananaConfig):
        super().__init__()
        self.config = config
        # Self-Attention
        self.w_Q = nn.Linear(config.latent_dim, config.latent_dim)
        self.w_K = nn.Linear(config.latent_dim, config.latent_dim)
        self.w_V = nn.Linear(config.latent_dim, config.latent_dim)
        self.softmax = nn.Softmax(-1)
        
        # FFNN
        self.FFNN = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.drop_out),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )

    def forward(self, z):
        # Self-Attention /w skip connection
        Q = self.w_Q(z)
        K = self.w_K(z)
        V = self.w_V(z)
        d_k = 1

        attn_score = self.softmax(Q @ K.permute(0, 2, 1)) / np.sqrt(d_k)
        z = attn_score @ V + z

        # FFNN /w skip connection
        z = self.FFNN(z) + z

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

        self.transformer_blocks = [TransformerBlock(config) 
                                   for _ in range(config.transformer_blocks)]

        self.output = nn.Linear(config.latent_dim, config.vocab_size)

    def forward(self, inputs):
        # Embedding
        # BxNx1 => BxNxD
        x = self.embedding(inputs)

        # Positional embedding

        # Transformer blocks 
        # BxNxD => BxNxD
        for block in self.transformer_blocks:
            x = block(x)

        # Output logits
        # BxNxD => BxNx(Vocab Size)
        logits = self.output(x)

        return logits