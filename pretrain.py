"""
Pretrain transformer with Masked LM and and different SOP variants
"""

import random
from random import randint, shuffle
from random import random as rand

import fire
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import models
import optim
import tokenization
import train
from utils import set_random_seed, get_gpu_or_cpu, get_random_word, truncate_tokens_pair


def get_random_offset(f, back_margin=2000):
    """
    seek random offset of file pointer
    """
    f.seek(0, 2)
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline()


class data_loader_for_sentence_pair():
    """
    Load sentence pair (sequential or random order) from corpus
    """

    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore')
        self.tokenize = tokenize
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """
        Method to read data for SOP
        :param f:
        :param length:
        :param discard_last_and_restart:
        :return:
        """
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line:  # end of file
                return None
            if not line.strip():  # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = []  # throw all and restart
                    continue
                else:
                    return tokens  # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def read_tokens_at_pos(self, f, length, pos, discard_last_and_restart=True):
        """
        Method to read data from CSV file for SOP1

        :param f:  file pointer
        :param length:  length of tokens to read
        :param pos: If need to skip any lines
        :param discard_last_and_restart: if strat reading from beginning
        :return:
        """

        # skipping lines for SOP1
        for x in range(pos):
            _skipping = f.readline()

        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line:  # end of file
                return None
            if not line.strip():  # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = []  # throw all and restart
                    continue
                else:
                    return tokens  # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __iter__(self):  # iterator to load data
        while True:
            batch = []
            print("-----SOP2-----")
            for i in range(self.batch_size):
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)

                tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
                """"
                1. Get which line to read from the current A line (eg 1st from A, 2nd from A, 3rd from A)
                    0 means 1st
                    1 means 2nd 
                    2 means 3rd
                    
                2. Get that line from the current A line
                
                3. is_next would be the relative position of line B from line A
                
                """
                possible_orders = [1, 2, 3, 4]
                line_b_rel_pos = random.choice(possible_orders)
                # set f_next to f_pos
                f_next = self.f_pos

                tokens_b = self.read_tokens_at_pos(f_next, len_tokens, line_b_rel_pos, False)

                if tokens_a is None or tokens_b is None:  # end of file
                    self.f_pos.seek(0, 0)  # reset file pointer
                    return
                instance = (line_b_rel_pos - 1, tokens_a, tokens_b)

                for proc in self.pipeline:
                    instance = proc(instance)

                batch.append(instance)

            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors


class Pipeline():
    """
    Pre-process Pipeline Class : callable
    """

    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Pretrain_transformer(Pipeline):
    """
    Pre-processing steps for pretraining transformer
    """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)


class Bert_model_for_pretraining(nn.Module):
    """
    Bert Model for Pretrain : Masked LM and SOP variants
    """

    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.transformer_block(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.manual_gelu_method
        self.norm = models.layer_norm(cfg)
        self.classifier = nn.Linear(cfg.dim, 4)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


def main(train_cfg='config/pretrain.json', model_cfg='config/bert_base.json', data_file='',
         model_file=None, data_parallel=True, vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/pretrain', log_dir='../exp/bert/pretrain/runs', max_len=512,
         max_pred=20, mask_prob=0.15):
    """
    :param train_cfg:  json file containig the pretraining params
    :param model_cfg: json file contianig the BERT model details
    :param data_file: data file containing the wikitext-103 unlabelled data
    :param model_file: model file if finetuning
    :param data_parallel: if running data parallel
    :param vocab: text file containng the vocab words
    :param save_dir: directory to save model checkpoints
    :param log_dir: directory to save tensorflow logs
    :param max_len: maximum sequence lenght
    :param max_pred: how many words to predict
    :param mask_prob: maxking probability
    """

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_random_seed(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Pretrain_transformer(max_pred,
                                     mask_prob,
                                     list(tokenizer.vocab.keys()),
                                     tokenizer.convert_tokens_to_ids,
                                     max_len)]
    data_iter = data_loader_for_sentence_pair(data_file, cfg.batch_size, tokenize, max_len, pipeline=pipeline)

    model = Bert_model_for_pretraining(model_cfg)
    # pretrain_file = '/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_sop2/logs/model_steps_240000.pt'   #checkpoint.load_model(model.transformer,"/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_with_sop/uncased_L-12_H-768_A-12/bert_model.ckpt")
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim_for_GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_gpu_or_cpu())
    writer = SummaryWriter(log_dir=log_dir)

    def Compute_combined_loss(model, batch, global_step):
        """
        Method to compute the overall loss
        """
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids)  # for masked LM
        loss_lm = (loss_lm * masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next)  # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                            },
                           global_step)
        return loss_lm + loss_clsf

    trainer.train(Compute_combined_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)
