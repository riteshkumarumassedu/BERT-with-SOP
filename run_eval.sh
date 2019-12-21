#!/bin/bash

export GLUE_DIR=/mnt/nfs/scratch1/riteshkumar/nlp_2019/glue_data
export BERT_PRETRAIN=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_with_sop/uncased_L-12_H-768_A-12
export SAVE_DIR=/mnt/nfs/scratch1/riteshkumar/nlp_code/bert_with_sop

python classify.py \
    --task mrpc \
    --mode eval \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/dev.tsv \
    --model_file model_steps_300.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512
