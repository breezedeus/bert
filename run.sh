#!/usr/bin/env bash

export BERT_BASE_DIR='/Users/king/Documents/Ein/语料/BERT/chinese_L-12_H-768_A-12'
TEST_DATA_DIR='./data'
DPF_DIR='/Users/king/Documents/Ein/Codes/dpf/data/projects/botlet_insurance/cv_0'
export BERT_TRAIN_SAMPLE_NUM=100
export BERT_DEV_SAMPLE_NUM=100  # dev 和 test 数据集都用这个值

## create pretraining data
#python create_pretraining_data.py \
#  --input_file=$TEST_DATA_DIR/test.txt \
#  --output_file=/tmp/tf_examples.tfrecord \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --do_lower_case=True \
#  --max_seq_length=128 \
#  --max_predictions_per_seq=20 \
#  --masked_lm_prob=0.15 \
#  --random_seed=12345 \
#  --dupe_factor=5


## run classifier on test data
#python run_classifier.py \
#  --task_name=test \
#  --do_train=true \
#  --do_eval=true \
#  --do_predict=true \
#  --data_dir=$TEST_DATA_DIR \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --max_seq_length=50 \
#  --train_batch_size=16 \
#  --learning_rate=5e-5 \
#  --num_train_epochs=200.0 \
#  --output_dir=$TEST_DATA_DIR/bert_classifier/

# run classifier on qabot data
python run_classifier.py \
  --task_name=qabot \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DPF_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=100 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=20.0 \
  --output_dir=$TEST_DATA_DIR/qabot_classifier/
