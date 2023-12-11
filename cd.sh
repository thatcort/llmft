export max_train_samples=128
export epochs=40
export warmup_ratio=0.5
export bsz=4
export num_gpus=1
export logging_steps=$((max_train_samples / (bsz * num_gpus)))
export learning_rate=1e-5
export model_name_or_path="facebook/opt-125m"

export HOME=/storage/ice1/6/5/afischer39/llmft
export CACHE_BASE_DIR=/storage/ice1/6/5/afischer39/llmft/cache
export OUTPUT_DIR=/storage/ice1/6/5/afischer39/llmft/logfiles
export HF_DATASETS_CACHE=/storage/ice1/6/5/afischer39/llmft/hf_datasets
export HF_EVALUATE_CACHE=/storage/ice1/6/5/afischer39/llmft/hf_evaluate
export HF_MODULES_CACHE=/storage/ice1/6/5/afischer39/llmft/hf_modules
export HF_MODELS_CACHE=/storage/ice1/6/5/afischer39/llmft/hf_lms

mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_EVALUATE_CACHE
mkdir -p $HF_MODULES_CACHE
mkdir -p $HF_MODELS_CACHE
mkdir -p $TORCH_EXTENSIONS_DIR


sh scripts/misc/setup.sh

python context_distillation.py \
        --output_dir $OUTPUT_DIR \
        --do_eval \
        --per_device_eval_batch_size 8 \
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
        --save_strategy no \
        --num_shots 10 \
        --separate_shots_by "\n\n" \
        --balanced \
        --shuffle \
        --data_seed 123 \
        --target_prefix " " \

