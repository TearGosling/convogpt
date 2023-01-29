#!/bin/bash

BASE_MODEL="EleutherAI/pythia-410m-deduped"
TRAIN_DATASET="train.jsonl"
EVAL_DATASET="eval.jsonl"
OUTPUT_DIR="models/pygmalion-350m"
EPOCHS=1
BATCH_SIZE=4
SAVE_STEPS=1000
LEARNING_RATE=3e-5

accelerate launch src/training/sft.py \
    --model $BASE_MODEL \
    --train_dataset $TRAIN_DATASET \
    --eval_dataset $EVAL_DATASET \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --learning_rate $LEARNING_RATE
