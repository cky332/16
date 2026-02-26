#!/bin/bash
# New task experiments for SWEET watermark failure case discovery
# Tests watermark on tasks with different code characteristics
#
# Priority (most likely to expose failures):
# 1. conala: single-line code, extremely short output
# 2. apps: different difficulty levels
# 3. ds1000 sub-libraries: API-heavy data science code
#
# Uses single process + device_map=auto to shard StarCoder across GPUs
# (avoids OOM from loading full model on a single 24GB GPU)

# Watermark parameters (matching actual experiment config)
gamma=0.5
delta=5.0
entropy_threshold=1.0

# ---- Task 1: CoNaLa (single-line code generation) ----
echo "=========================================="
echo "Running SWEET experiment for: CoNaLa"
echo "=========================================="
accelerate launch --num_processes 1 main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task conala \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size 20 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 10 \
    --max_length_generation 256 \
    --save_generations \
    --outputs_dir outputs/sweet_conala \
    --sweet \
    --gamma $gamma \
    --delta $delta \
    --entropy_threshold $entropy_threshold \
    --device_map auto \
    --generation_only

# ---- Task 2: APPS (different difficulty levels) ----
for level in introductory interview competition; do
    echo "=========================================="
    echo "Running SWEET experiment for: APPS-${level}"
    echo "=========================================="
    accelerate launch --num_processes 1 main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task "apps-${level}" \
        --temperature 0.2 \
        --precision bf16 \
        --batch_size 5 \
        --allow_code_execution \
        --do_sample \
        --top_p 0.95 \
        --n_samples 10 \
        --max_length_generation 2048 \
        --save_generations \
        --outputs_dir "outputs/sweet_apps_${level}" \
        --sweet \
        --gamma $gamma \
        --delta $delta \
        --entropy_threshold $entropy_threshold \
        --device_map auto \
        --generation_only
done

# ---- Task 3: DS-1000 sub-libraries ----
for lib in numpy pandas scipy matplotlib sklearn tensorflow pytorch; do
    echo "=========================================="
    echo "Running SWEET experiment for: DS-1000 ${lib}"
    echo "=========================================="
    accelerate launch --num_processes 1 main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task "ds1000-${lib}-completion" \
        --temperature 0.2 \
        --precision bf16 \
        --batch_size 10 \
        --allow_code_execution \
        --do_sample \
        --top_p 0.5 \
        --n_samples 10 \
        --max_length_generation 2048 \
        --save_generations \
        --outputs_dir "outputs/sweet_ds1000_${lib}" \
        --sweet \
        --gamma $gamma \
        --delta $delta \
        --entropy_threshold $entropy_threshold \
        --device_map auto \
        --generation_only
done

# ---- Task 4: MBPP ----
echo "=========================================="
echo "Running SWEET experiment for: MBPP"
echo "=========================================="
accelerate launch --num_processes 1 main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task mbpp \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size 5 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 10 \
    --max_length_generation 2048 \
    --save_generations \
    --outputs_dir outputs/sweet_mbpp \
    --sweet \
    --gamma $gamma \
    --delta $delta \
    --entropy_threshold $entropy_threshold \
    --device_map auto \
    --generation_only

echo ""
echo "All new task generation experiments complete."
