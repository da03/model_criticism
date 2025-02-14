# Model Criticism for Text Generation

The code is adapted from HuggingFace transformers.

## Prerequisites

The code has been tested on Python 3.8. In addition, we need to install `transformers` included in this repo by

```
pip install sklearn
pip install datasets
pip install --editable .
```


## Data

```
wget http://52.24.243.180/static/data.tgz
tar zxf data.tgz
```

## Usage

### Train LMs

To train a language model in the W/O Title setting, use the below command:

```
export MODEL=EleutherAI/gpt-neo-125M # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=30
export TRAIN_BATCH_SIZE=4 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=2 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=8 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=1
export SAVE_FOLDER=language_model_checkpoints/GPT-Neo/Without_Title
export TRAIN_FILE=data/train.wo_title.txt
export TEST_FILE=data/val.wo_title.txt
python examples/pytorch/language-modeling/run_clm.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --per_device_eval_batch_size=${EVAL_BATCH_SIZE} \
       --gradient_accumulation_steps=${ACCUMULATION} \
       --output_dir=${SAVE_FOLDER} \
       --model_type=${MODEL} \
       --model_name_or_path=${MODEL} \
       --do_train \
       --do_eval \
       --train_file=${TRAIN_FILE} \
       --validation_file=${TEST_FILE} --overwrite_output_dir --save_total_limit=${SAVE_TOTAL_LIMIT} \
       --learning_rate=${LR} --num_train_epochs=${EPOCHS} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch \
       --block_size 2048 > log.trainLM.wo_title.gptneo 2>&1&

```

(It would be nice if you have bandwidth for running the below setting, if not please just ignore the below setting). To train a language model in the W/ Title setting, use the below command:

```
export MODEL=EleutherAI/gpt-neo-125M # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=30
export TRAIN_BATCH_SIZE=4 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=2 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=8 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=1
export SAVE_FOLDER=language_model_checkpoints/GPT-Neo/With_Title
export TRAIN_FILE=data/train.w_title.txt
export TEST_FILE=data/val.w_title.txt
python examples/pytorch/language-modeling/run_clm.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --per_device_eval_batch_size=${EVAL_BATCH_SIZE} \
       --gradient_accumulation_steps=${ACCUMULATION} \
       --output_dir=${SAVE_FOLDER} \
       --model_type=${MODEL} \
       --model_name_or_path=${MODEL} \
       --do_train \
       --do_eval \
       --train_data_file=${TRAIN_FILE} \
       --eval_data_file=${TEST_FILE} --overwrite_output_dir --save_total_limit=${SAVE_TOTAL_LIMIT} \
       --learning_rate=${LR} --num_train_epochs=${EPOCHS} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch \
       --block_size 2048 > log.trainLM.w_title.gptneo 2>&1&

```

### Generate from LMs

I forgot to ask during our meeting: do you plan to share the trained LM after training? I'm not sure if I have the infrastructure for sampling either... Maybe we can discuss after training?

To generate, for the W/O Title setting (if 10k samples is too much, I think 1k is good as well by using `--num_samples 1000`):

```
export SAVE_FOLDER=language_model_checkpoints/GPT-Neo/Without_Title
stdbuf -oL python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/GPT-Neo/Without_Title \
        --output_file generation.wo_title.gpt2.jsonl \
        --num_samples 10000 > log.gen.wo_title.gptneo 2>&1&
```

(It would be nice if you have bandwidth for running the below setting, if not please just ignore the below setting). For the W/ Title setting (if 10k samples is too much, I think 1k is good as well by using `--num_samples 1000`):

```
export SAVE_FOLDER=language_model_checkpoints/GPT-Neo/With_Title
python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/GPT-Neo/With_Title \
        --output_file generation.w_title.gpt2.jsonl \
        --num_samples 10000 > log.gen.w_title.gptneo 2>&1&
```
