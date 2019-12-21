"""
Module to support the downstream classification tasks
"""
import fire
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tokenization
import models
import optim
import train
from data_reader_helper import *
from data_tokenizer import *
from utils import set_random_seed, get_gpu_or_cpu

class Classifier(nn.Module):
    """
    Classifier with Transformer
    """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.transformer_block(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits

def main(task='mrpc', train_cfg='config/train_mrpc.json', model_cfg='config/bert_base.json',
         data_file='../glue/MRPC/train.tsv', model_file=None, pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
         data_parallel=True, vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/mrpc', max_len=128, mode='train'):
    """
    :param task:            dataset for which you want to run
    :param train_cfg:       json file containing params for the classification run
    :param model_cfg:       json file containing the details about BERT baase model
    :param data_file:       csv file realted the the data set
    :param model_file:
    :param pretrain_file:   pretrained model weights checkpoint
    :param data_parallel:   if we want to run data parallel
    :param vocab:           vocab file "uses the vocab file which came along with the bet uncased weights"
    :param save_dir:        path to save the checkpoints
    :param max_len:         maximum sequence length
    :param mode:            train, validation or test
    :return:
    """

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    set_random_seed(cfg.seed)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_to_class_mapping(task)
    pipeline = [Tokenize_data(tokenizer.convert_to_unicode, tokenizer.tokenize),
                Tokenizer_helper(max_len),
                Indexing_the_tokens(tokenizer.convert_tokens_to_ids,
                                    TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()

    trainer = train.Trainer(cfg, model, data_iter, optim.optim_for_GPU(cfg, model), save_dir, get_gpu_or_cpu())

    if mode == 'train':
        def get_loss(model, batch, global_step):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_id)
            return loss

        trainer.train(get_loss, model_file, pretrain_file, data_parallel)

    elif mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)
        total_accuracy = torch.cat(results).mean().item()
        print('Accuracy:', total_accuracy)


if __name__ == '__main__':
    fire.Fire(main)
