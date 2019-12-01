"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""
import random
from random import randint, shuffle
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter

import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair, _sample_mask

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence

class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample
        # self.f_tmpline = open(file, "r", encoding='utf-8', errors='ignore') # for a negative (random) sample
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens


    def read_tokens_at_pos(self, f, length, pos, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """

        # skipping lines
        for x in range(pos-1):
            _skipping = f.readline()

        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                # ALBERT is same  randomly generate input
                # sequences shorter than 512 with a probability of 10%.
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)

                tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
                # seek_random_offset(self.f_neg)
                # f_next = self.f_pos if is_next else self.f_neg
                # f_next = self.f_pos # `f_next` should be next point


                """"
                1. Get which line to read from the current A line (eg 1st from A, 2nd from A, 3rd from A)
                    0 means 1st
                    1 means 2nd 
                    2 means 3rd
                    
                2. Get that line from the current A line
                
                3. is_next would be the relative position of line B from line A
                
                """
                possible_orders = [1,2,3]
                line_b_rel_pos = random.choice(possible_orders)
                # set f_next to f_pos
                f_next = self.f_pos



                tokens_b = self.read_tokens_at_pos(f_next, len_tokens,line_b_rel_pos, False)

                if tokens_a is None or tokens_b is None: # end of file
                    self.f_pos.seek(0, 0) # reset file pointer
                    return

                # SOP, sentence-order prediction
                instance = (line_b_rel_pos-1, tokens_a, tokens_b)

                for proc in self.pipeline:
                    instance = proc(instance)

                batch.append(instance)

            # To Tensor
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len,
                 mask_alpha, mask_beta, max_gram):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words

        self.indexer = indexer # function from token to token index
        self.max_len = max_len
        self.mask_alpha = mask_alpha
        self.mask_beta = mask_beta
        self.max_gram = max_gram

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))

        # For masked Language Models
        masked_tokens, masked_pos, tokens = _sample_mask(tokens, self.mask_alpha,
                                            self.mask_beta, self.max_gram,
                                            goal_num_predict=n_pred)

        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > len(masked_ids):
            masked_ids.extend([0] * (self.max_pred - len(masked_ids)))
        if self.max_pred > len(masked_pos):
            masked_pos.extend([0] * (self.max_pred - len(masked_pos)))
        if self.max_pred > len(masked_weights):
            masked_weights.extend([0] * (self.max_pred - len(masked_weights)))

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)

class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 3)   # to support upto 3 lines order

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        ## project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf

def main(args):

    cfg = train.Config.from_json(args.train_cfg)
    model_cfg = models.Config.from_json(args.model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(args.max_pred,
                                    args.mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    model_cfg.max_len,
                                    args.mask_alpha,
                                    args.mask_beta,
                                    args.max_gram)]
    data_iter = SentPairDataLoader(args.data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   model_cfg.max_len,
                                   pipeline=pipeline)

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, args.save_dir, get_device())

    writer = SummaryWriter(log_dir=args.log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_sop = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_sop': loss_sop.item(),
                            'loss_total': (loss_lm + loss_sop).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)
        return loss_lm + loss_sop

    trainer.train(get_loss, model_file=None, data_parallel=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ALBERT Language Model')
    parser.add_argument('--data_file', type=str, default='./data/wiki.train.tokens')
    parser.add_argument('--vocab', type=str, default='./data/vocab.txt')
    parser.add_argument('--train_cfg', type=str, default='./config/pretrain.json')
    parser.add_argument('--model_cfg', type=str, default='./config/albert_unittest.json')

    # official google-reacher/bert is use 20, but 20/512(=seq_len)*100 make only 3% Mask
    # So, using 76(=0.15*512) as `max_pred`
    parser.add_argument('--max_pred', type=int, default=76, help='max tokens of prediction')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='masking probability')

    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    parser.add_argument('--mask_alpha', type=int,
                        default=4, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--max_gram', type=int,
                        default=3, help="number of max n-gram to masking")

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    main(args=args)
