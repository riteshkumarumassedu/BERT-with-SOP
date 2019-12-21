#!/bin/bash

export GLUE_DIR=/mnt/nfs/scratch1/riteshkumar/nlp_2019/glue_data
export BERT_PRETRAIN=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_sop2/uncased_L-12_H-768_A-12
export SAVE_DIR=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_sop2

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
