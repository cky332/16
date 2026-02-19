# detection phase (EXP-edit)

# ---- 选择任务: 取消注释你需要的任务，注释掉其余的 ----

# for humaneval
task="humaneval"
max_len=512
batch_size=20
top_p=0.95
n_sample=40

# # for mbpp
# task="mbpp"
# max_len=2048
# batch_size=5
# top_p=0.95
# n_sample=20

# # for ds1000
# task="ds1000-all-completion"
# max_len=1024
# batch_size=10
# top_p=0.5
# n_sample=40

# ---- 水印参数 ----
key_length=512
n_runs=100

accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task $task \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size $batch_size \
    --allow_code_execution \
    --do_sample \
    --top_p $top_p \
    --n_samples $n_sample \
    --max_length_generation $max_len \
    --load_generations_path outputs/exp_${task}/generations.json \
    --outputs_dir outputs/exp_${task} \
    --exp \
    --key_length $key_length \
    --n_runs $n_runs
