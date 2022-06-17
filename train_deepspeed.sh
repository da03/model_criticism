export MODEL_BASE=gpt-neo-125M # either gpt-neox-20b or gpt-neo-2.7B would be nice
export MODEL=EleutherAI/${MODEL_BASE} # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=30
export TRAIN_BATCH_SIZE=1 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=1 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=1 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=10
export SAVE_FOLDER=language_model_checkpoints/${MODEL_BASE}/Without_Title
export TRAIN_FILE=data/train.wo_title.txt
export TEST_FILE=data/val.wo_title.txt
deepspeed --num_gpus=8 --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm.py \
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
       --block_size 2048 > log.trainLM.wo_title.${MODEL_BASE}.block2048.deepspeed 2>&1&
export MODEL_BASE=gpt-neo-1.3B # either gpt-neox-20b or gpt-neo-2.7B would be nice
export MODEL=EleutherAI/${MODEL_BASE} # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=30
export TRAIN_BATCH_SIZE=1 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=1 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=1 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=10
export SAVE_FOLDER=language_model_checkpoints/${MODEL_BASE}/Without_Title
export TRAIN_FILE=data/train.wo_title.txt
export TEST_FILE=data/val.wo_title.txt
deepspeed --num_gpus=8 --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm.py \
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
       --block_size 2048 --gradient_checkpointing > log.trainLM.wo_title.${MODEL_BASE}.block2048.deepspeed.gradient_checkpoint 2>&1&
export MODEL_BASE=gpt-neo-2.7B # either gpt-neox-20b or gpt-neo-2.7B would be nice
export MODEL=EleutherAI/${MODEL_BASE} # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=3
export TRAIN_BATCH_SIZE=1 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=1 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=1 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=10
export SAVE_FOLDER=language_model_checkpoints/${MODEL_BASE}/Without_Title
export TRAIN_FILE=data/train.wo_title.txt
export TEST_FILE=data/val.wo_title.txt
/home/mchorse/anaconda3/envs/yuntian/bin/deepspeed --num_gpus=8 --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --deepspeed=ds_config_gptneo.json \
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
       --block_size 2048 --fp16  > log.trainLM.wo_title.${MODEL_BASE}.block2048.deepspeed.fp16.withconfig 2>&1&
export MODEL_BASE=gpt-neo-2.7B # either gpt-neox-20b or gpt-neo-2.7B would be nice
export MODEL=EleutherAI/${MODEL_BASE} # either gpt-neox-20b or gpt-neo-2.7B would be nice
export LR=5e-5
export EPOCHS=30
export TRAIN_BATCH_SIZE=1 # batch size for training, this might need to be set to smaller values if necessary?
export ACCUMULATION=1 # accumulation steps. The effective total batch size should be 8. For multi-GPU training maybe this should be set to 1?
export EVAL_BATCH_SIZE=1 # batch size for evaluation, this might need to be set to a smaller value.
export SAVE_TOTAL_LIMIT=10
export SAVE_FOLDER=language_model_checkpoints/${MODEL_BASE}/Without_Title
export TRAIN_FILE=data/train.wo_title.txt
export TEST_FILE=data/val.wo_title.txt
/home/mchorse/anaconda3/envs/yuntian/bin/deepspeed --num_gpus=8 --hostfile=hostfile --master_port=54321 examples/pytorch/language-modeling/run_clm.py \
       --per_device_train_batch_size=${TRAIN_BATCH_SIZE} \
       --deepspeed=ds_config_gptneo.json \
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
       --block_size 2048 --fp16  > log.trainLM.wo_title.${MODEL_BASE}.block2048.deepspeed.fp16.withconfig 2>&1&
