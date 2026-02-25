#!/bin/bash
# Robustness testing: detect watermarks on code-transformed generations
# First generates transformed code using code_transform.py, then runs detection
#
# Tests whether simple code transformations can break watermark detection:
# - Variable renaming
# - Comment removal/addition
# - Whitespace normalization
# - Combined heavy transformation

# Watermark parameters
gamma=0.5
delta=5.0
entropy_threshold=1.0

# Step 1: Generate transformations (no GPU needed)
echo "=========================================="
echo "Step 1: Generating code transformations"
echo "=========================================="
python code_transform.py \
    --input outputs_sweet/generations.json \
    --output_dir transformed_generations

# Step 2: Run detection on each transformed version
transforms="rename_vars remove_comments add_comments normalize_ws add_blanks remove_blanks combined_light combined_heavy"

for transform in $transforms; do
    gen_file="transformed_generations/generations_${transform}.json"
    output_dir="outputs/sweet_robustness_${transform}"

    if [ ! -f "$gen_file" ]; then
        echo "Skipping ${transform}: file not found"
        continue
    fi

    echo "=========================================="
    echo "Running detection on: ${transform}"
    echo "=========================================="
    # Use single process + device_map=auto to shard model across GPUs
    # (evaluation-only mode loads model on main process only, so multi-process
    #  would OOM trying to fit the full model on a single GPU)
    accelerate launch --num_processes 1 main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task humaneval \
        --temperature 0.2 \
        --precision bf16 \
        --batch_size 20 \
        --allow_code_execution \
        --do_sample \
        --top_p 0.95 \
        --n_samples 40 \
        --max_length_generation 512 \
        --load_generations_path $gen_file \
        --outputs_dir $output_dir \
        --sweet \
        --gamma $gamma \
        --delta $delta \
        --entropy_threshold $entropy_threshold \
        --device_map auto
done

echo ""
echo "Robustness testing complete."
echo "Compare detection rates across transformations to assess watermark robustness."
