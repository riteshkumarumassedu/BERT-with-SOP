"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Utils Functions """

import os
import random
import logging

import numpy as np
import torch


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def find_sublist(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    https://codereview.stackexchange.com/questions/19627/finding-sub-list
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger

def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False

def _sample_mask(seg, mask_alpha, mask_beta,
                 max_gram=3, goal_num_predict=85):
    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    # 3-gram implementation
    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True) # p(n) = 1/n / sigma(1/k)

    cur_len = 0

    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)

        # `mask_alpha` : number of tokens forming group
        # `mask_beta` : number of tokens to be masked in each groups.
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx

        while beg < seg_len and not _is_start_piece([seg[beg]]):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece([seg[beg]]):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    tokens, masked_tokens, masked_pos = [], [], []
    for i in range(seg_len):
        if mask[i] and (seg[i] != '[CLS]' and seg[i] != '[SEP]'):
            masked_tokens.append(seg[i])
            masked_pos.append(i)
            tokens.append('[MASK]')
        else:
            tokens.append(seg[i])
    return masked_tokens, masked_pos, tokens
