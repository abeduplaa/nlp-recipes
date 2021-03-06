# Script for running distributed training

[main]

#home directory:
home_dir = /home/ubuntu/mnt/training_results/

# Directory to cache the tokenizer. 
CACHE_DIR= %(home_dir)s/tokenizer_cache/

# Directory to download the preprocessed data.
data_dir= %(home_dir)s/downloaded_data/

# Directory to save the output model and prediction results.
output_dir= %(home_dir)s/output/

# Transformer model used in the extractive summarization
model_name=bert-base-german-dbmdz-uncased

# batch size in terms of the number of samples in training
batch_size=10

# Maximum number of training steps run in training. If quick_run is set, this is ignored
max_steps= 100000

# Warm-up number of training steps run in training. If quick_run is set, this is ignored
warmup_steps=5000

# Number of sentences selected in prediction for evaluation.
top_n=3

# Summary file name generated by prediction for evaluation.
summary_filename=

# training data file which is saved through torch 
train_file= 

# test data file for evaluation.
test_file=

# Whether to have a quick run
quick_run= 

# Encoder types in the extractive summarizer. —> ["baseline", "classifier", "transformer", "rnn"],
encoder=

# training learning rate
learning_rate=

# model file name saved for evaluation.
model_filename=


export OUTPUT_DIR_NAME=bart_utest_output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=cnn_tiny/ \
--model_name_or_path=sshleifer/bart-tiny-random \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=1  \
--gpus=0 \
--do_train $@

rm -rf cnn_tiny
rm -rf $OUTPUT_DIR