#!/bin/bash
# Parameter sensitivity analysis for SWEET watermark
# Sweeps over key parameters to understand their impact on detection
#
# Parameters tested:
# 1. entropy_threshold: controls which tokens get watermarked
# 2. temperature: controls code generation randomness
# 3. delta: controls watermark strength

task="humaneval"
max_len=512
batch_size=20
top_p=0.95
n_sample=40
gamma=0.5

# ---- Sweep 1: entropy_threshold ----
echo "===== SWEEP: entropy_threshold ====="
delta=5.0
temperature=0.2

for thresh in 0.1 0.3 0.5 0.8 1.0 1.5 2.0; do
    echo "--- entropy_threshold=${thresh} ---"
    output_dir="outputs/sweet_sweep_thresh_${thresh}"

    accelerate launch main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task $task \
        --temperature $temperature \
        --precision bf16 \
        --batch_size $batch_size \
        --allow_code_execution \
        --do_sample \
        --top_p $top_p \
        --n_samples $n_sample \
        --max_length_generation $max_len \
        --save_generations \
        --outputs_dir $output_dir \
        --sweet \
        --gamma $gamma \
        --delta $delta \
        --entropy_threshold $thresh \
        --generation_only
done

# ---- Sweep 2: temperature ----
echo "===== SWEEP: temperature ====="
delta=5.0
entropy_threshold=1.0

for temp in 0.2 0.4 0.6 0.8 1.0; do
    echo "--- temperature=${temp} ---"
    output_dir="outputs/sweet_sweep_temp_${temp}"

    accelerate launch main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task $task \
        --temperature $temp \
        --precision bf16 \
        --batch_size $batch_size \
        --allow_code_execution \
        --do_sample \
        --top_p $top_p \
        --n_samples $n_sample \
        --max_length_generation $max_len \
        --save_generations \
        --outputs_dir $output_dir \
        --sweet \
        --gamma $gamma \
        --delta $delta \
        --entropy_threshold $entropy_threshold \
        --generation_only
done

# ---- Sweep 3: delta (watermark strength) ----
echo "===== SWEEP: delta ====="
entropy_threshold=1.0
temperature=0.2

for d in 0.5 1.0 2.0 5.0 10.0; do
    echo "--- delta=${d} ---"
    output_dir="outputs/sweet_sweep_delta_${d}"

    accelerate launch main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task $task \
        --temperature $temperature \
        --precision bf16 \
        --batch_size $batch_size \
        --allow_code_execution \
        --do_sample \
        --top_p $top_p \
        --n_samples $n_sample \
        --max_length_generation $max_len \
        --save_generations \
        --outputs_dir $output_dir \
        --sweet \
        --gamma $gamma \
        --delta $d \
        --entropy_threshold $entropy_threshold \
        --generation_only
done

echo ""
echo "All parameter sweep experiments complete."
echo "Analyze results by comparing detection rates across parameter values."
