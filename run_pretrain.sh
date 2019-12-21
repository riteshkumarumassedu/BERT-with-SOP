#!/bin/bash

export DATA_FILE=/mnt/nfs/scratch1/riteshkumar/nlp_2019/wikitext-103/wiki.train.tokens
export BERT_PRETRAIN=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_with_sop/uncased_L-12_H-768_A-12
export SAVE_DIR=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_sop2/without_clsf

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
