docker build -f ./docker/Dockerfile \
    --build-arg USER_UID=$UID \
    --build-arg USER_NAME=$(id -un) \
    -t llmft:bc .

docker run -it --rm --gpus=all --pid=host --ipc=host --user brian \
    -v .:/llmft \
    -v ./datasets:/datasets \
    -v ./logfiles:/logfiles \
    -v ./.cache:/cache \
    llmft:bc

export PROJECT_DIR=/llmft
source /llmft/scripts/misc/setup.sh

# bash /llmft/scripts/in_context/mnli/run_minimal.sh mnli 2 EleutherAI/pythia-1.4b 1 60000









#################### Context Distillation #########

export max_train_samples=128
export epochs=40
export warmup_ratio=0.5
export bsz=4
export num_gpus=1
export logging_steps=$((max_train_samples / (bsz * num_gpus)))
export learning_rate=1e-5
export model_name_or_path="facebook/opt-125m"
python $PROJECT_DIR/context_distillation.py \
        --output_dir $OUTPUT_DIR \
        --do_eval \
        --per_device_eval_batch_size 10 \
        --fp16 \
        --seed 0 \
        --data_seed 0 \
        --report_to "none" \
        --model_name_or_path $model_name_or_path \
        --cache_dir $HF_MODELS_CACHE \
        --task_name "mnli" \
        --dataset_cache_dir $HF_DATASETS_CACHE \
        --overwrite_cache True \
        --pattern "{text1} {text2} ?" \
        --target_tokens "ĠYes,ĠNo" \
        --max_seq_length 2048 \
        --target_tokens "ĠYes,ĠNo" \
        --max_seq_length 256 \
        --overwrite_output_dir \
        --do_train \
        --max_train_samples $max_train_samples \
        --num_train_epochs $epochs \
        --warmup_ratio $warmup_ratio \
        --logging_first_step true \
        --logging_steps $logging_steps \
        --learning_rate $learning_rate \
        --weight_decay 0.0 \
        --evaluation_strategy epoch \
        --per_device_eval_batch_size 10 \
        --eval_on_hans \
        --save_strategy no

################# Fine Tuning: ##################
export max_train_samples=128
export epochs=40
export warmup_ratio=0.5
export bsz=4
export num_gpus=1
export logging_steps=$((max_train_samples / (bsz * num_gpus)))
export learning_rate=1e-5
export model_name_or_path="facebook/opt-350m"
python $PROJECT_DIR/fine_tuning.py \
        --output_dir $OUTPUT_DIR \
        --do_eval \
        --per_device_eval_batch_size 10 \
        --fp16 \
        --seed 0 \
        --data_seed 0 \
        --report_to "none" \
        --model_name_or_path $model_name_or_path \
        --cache_dir $HF_MODELS_CACHE \
        --task_name "mnli" \
        --dataset_cache_dir $HF_DATASETS_CACHE \
        --overwrite_cache True \
        --pattern "{text1} {text2} ?" \
        --target_tokens "ĠYes,ĠNo" \
        --max_seq_length 2048 \
        --target_tokens "ĠYes,ĠNo" \
        --max_seq_length 256 \
        --overwrite_output_dir \
        --do_train \
        --max_train_samples $max_train_samples \
        --num_train_epochs $epochs \
        --warmup_ratio $warmup_ratio \
        --logging_first_step true \
        --logging_steps $logging_steps \
        --learning_rate $learning_rate \
        --weight_decay 0.0 \
        --evaluation_strategy epoch \
        --per_device_eval_batch_size 10 \
        --eval_on_hans \
        --save_strategy no














        