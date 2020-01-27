#!/bin/bash

export GLUE_DIR=glue_data_path
export BERT_PRETRAIN=uncased_L-12_H-768_A-12
export SAVE_DIR=dir to save results

python classify.py \
    --task mrpc \
    --mode train \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/train.tsv \
    --pretrain_file fine_tune240/model_steps_140000.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 128
