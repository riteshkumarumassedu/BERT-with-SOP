"""
Contains the implementation of entire BERT model different modules
Derived from Hugging Face implementation
"""

import math
import json
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last
import yaml
from yaml import Loader

class Config(NamedTuple):

    """
    Load the BERT model parameter details from the config file
    """
    with open("bert_config.yml", 'r') as config_file:
        bert_config = yaml.load(config_file, Loader=Loader)

    vocab_size: int = None
    dim: int = bert_config['dim']
    n_layers: int = bert_config['n_layers']
    n_heads: int = bert_config['n_heads']
    dim_ff: int = bert_config['dim_ff']
    p_drop_hidden: float = bert_config['drop_hidden']
    p_drop_attn: float = bert_config['drop_attn']
    max_len: int = bert_config['max_len']
    n_segments: int = bert_config['n_segments']

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def manual_gelu_method(x):
    """
    Implementation of the gelu activation function by Hugging Face
    :param x:
    :return:
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class embedding_class(nn.Module):
    """
    The embedding module from word, position and token_type embeddings
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)
        self.norm = layer_norm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))

class transformer_block(nn.Module):
    """
    Transformer with Self-attention blocks
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed = embedding_class(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h

class multi_headed_attn(nn.Module):
    """
    Multi-Headed Dot Product Attention
    """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)  for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h

class layer_norm(nn.Module):

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class position_wise_feed_forward(nn.Module):
    """
    Position wise Feed Forward bloack
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        return self.fc2(manual_gelu_method(self.fc1(x)))


class Block(nn.Module):
    """
     Transformer Block
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn = multi_headed_attn(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = layer_norm(cfg)
        self.pwff = position_wise_feed_forward(cfg)
        self.norm2 = layer_norm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h



