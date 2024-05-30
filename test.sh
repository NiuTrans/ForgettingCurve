
# model_id="meta-llama/Llama-2-7b-hf"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=5,6,7 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 4500 \
#     --training_len 4096 \
#     --save_path "." \
#     --device "cuda"



# model_id="meta-llama/Llama-2-7b-chat-hf"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=5,6,7 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 4500 \
#     --training_len 4096 \
#     --save_path "." \
#     --device "cuda"

# model_id="mistralai/Mistral-7B-v0.1"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 33000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"

# model_id="mistralai/Mistral-7B-Instruct-v0.2"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 33000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"



# model_id="togethercomputer/LLaMA-2-7B-32K"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 33000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"


# model_id="NousResearch/Yarn-Llama-2-7b-128k"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 128000 \
#     --training_len 128000 \
#     --save_path "." \
#     --device "cuda"


# model_id="Yukang/Llama-2-7b-longlora-100k-ft"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 100000 \
#     --training_len 100000 \
#     --save_path "." \
#     --device "cuda"


# model_id="THUDM/chatglm3-6b-32k"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 33000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"


# model_id="Qwen/Qwen1.5-7B"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 33000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"


# model_id="google/gemma-7b"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 8500 \
#     --training_len 8192 \
#     --save_path "." \
#     --device "cuda"

# model_id="alpindale/Mistral-7B-v0.2-hf"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 64000 \
#     --training_len 32768 \
#     --save_path "." \
#     --device "cuda"

# model_id="01-ai/Yi-6B-200K"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 200000 \
#     --training_len 200000 \
#     --save_path "." \
#     --device "cuda"



# model_id="internlm/internlm2-base-7b"
# title=$(basename "$model_id" | cut -d'/' -f2)
# python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 200000 \
#     --training_len 200000 \
#     --save_path "." \
#     --device "cuda"




# model_id="state-spaces/mamba-2.8b-hf"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=7 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 16000 \
#     --training_len 2048 \
#     --save_path "." \
#     --device "cuda"





# model_id="RWKV/v5-Eagle-7B-HF"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=7 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 16000 \
#     --training_len 4096 \
#     --save_path "." \
#     --device "cuda"


# model_id="ai21labs/Jamba-v0.1"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 265 \
#     --training_len 256000 \
#     --save_path "." \
#     --device "cuda"

# model_id="fla-hub/gla-1.3B-100B"
# title=$(basename "$model_id" | cut -d'/' -f2)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python draw.py \
#     --model_id $model_id \
#     --title $title \
#     --repeat_time 10 \
#     --granularity 32 \
#     --granularity_type "linear" \
#     --data_type "order" \
#     --test_max_length 8000 \
#     --training_len 2048 \
#     --save_path "." \
#     --device "cuda"


model_id="meta-llama/Meta-Llama-3-8B"
title=$(basename "$model_id" | cut -d'/' -f2)
CUDA_VISIBLE_DEVICES=0,1,2,3 python draw.py \
    --model_id $model_id \
    --title $title \
    --repeat_time 10 \
    --granularity 32 \
    --granularity_type "linear" \
    --data_type "order" \
    --test_max_length 16000 \
    --training_len 8000 \
    --save_path "." \
    --device "cuda"