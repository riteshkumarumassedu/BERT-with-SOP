#!/bin/bash

export DATA_FILE=wikitext-103/wiki.train.tokens
export BERT_PRETRAIN=uncased_L-12_H-768_A-12
export SAVE_DIR=dir to save results

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
