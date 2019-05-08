###################################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月13日 星期三 10时31分56秒
#=============================================================
#!/bin/bash
source activate tensorflow_new_3.6
export BERT_BASE_DIR=/root/zhaomeng/baidu_ERNIE/pad_to_tf/checkpoints
#export BERT_BASE_DIR=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12

python run_sequencelabeling.py --task_name=test  \
                              --output_dir=./output \
                              --data_dir=./data \
                              --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
                              --bert_config_file=$BERT_BASE_DIR/bert_config.json \
                              --vocab_file=$BERT_BASE_DIR/vocab.txt \
                              --max_seq_length=64  \
                              --do_train=true \
                              --num_train_epochs=10 \
                              --learning_rate=5e-5  \
                              --train_batch_size=64 

