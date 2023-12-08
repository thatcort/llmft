#!/bin/bash
#SBATCH -JGPUExample                                # Job name
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=1   # Number of nodes, GPUs, and cores required
#SBATCH --mem-per-gpu=20G                           # Memory per gpu
#SBATCH -t15                                        # Duration of the job (Ex: 15 mins)
#SBATCH -oreports/eval-report-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=afischer39@gatech.edu            # e-mail address for notifications

conda activate myenv

srun python eval.py \
        --output_dir $OUTPUT_DIR \
        --do_eval \
        --per_device_eval_batch_size 10 \
        --fp16 \
        --seed 0 \
        --data_seed 0 \
        --report_to "none" \
        --model_name_or_path "facebook/opt-125m" \
        --cache_dir $HF_MODELS_CACHE \
        --task_name "mnli" \
        --dataset_cache_dir $HF_DATASETS_CACHE \
        --eval_task_name "hans" \
        --pattern "{text1} {text2} ?" \
        --target_prefix " " \
        --target_tokens "ĠYes,ĠNo" \
        --separate_shots_by "\n\n" \
        --group "minimal" \
        --max_seq_length 2048 \
        --num_shots 2 \
        --balanced \
        --shuffle
        --gpu 1000  # Run test example