#!/bin/bash
# Multi-language SWEET watermark experiments using MultiPL-E benchmark
# Tests watermark detectability across different programming languages
#
# Hypothesis: Static-typed languages (Java, C++, Go) have more deterministic
# tokens (type declarations, boilerplate), leading to lower watermarking_fraction
# and worse detection rates compared to Python.

# Watermark parameters (matching actual experiment config)
gamma=0.5
delta=5.0
entropy_threshold=1.0

# Generation parameters
temperature=0.2
top_p=0.95
n_sample=10       # reduced for multi-language sweep
batch_size=10
max_len=512

# Languages to test (representative subset covering different paradigms)
# - cpp: static types, templates, heavy boilerplate
# - java: very verbose, lots of keywords
# - js: dynamic, less boilerplate than Java
# - go: strict error handling patterns
# - rs: ownership system, complex type annotations
languages="cpp java js go rs"

for lang in $languages; do
    echo "=========================================="
    echo "Running SWEET experiment for: multiple-${lang}"
    echo "=========================================="

    output_dir="outputs/sweet_multiple_${lang}"

    # Generation phase
    accelerate launch main.py \
        --model bigcode/starcoder \
        --use_auth_token \
        --task "multiple-${lang}" \
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
        --entropy_threshold $entropy_threshold \
        --generation_only

    echo "Generation complete for ${lang}, output in ${output_dir}"
done

echo ""
echo "All multi-language generation experiments complete."
echo "To run detection, use run_sweet_multilang_detection.sh"
