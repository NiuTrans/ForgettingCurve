from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

# 获取特定环境变量的值
hf_api_token = os.environ.get("HF_API_TOKEN")

# 如果环境变量存在，打印它的值
if hf_api_token:
    print(f"HF_API_TOKEN: {hf_api_token}")
else:
    print("HF_API_TOKEN 环境变量未设置。")

access_token = hf_api_token

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="cpu", torch_dtype=torch.float16).eval()
# model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", device_map="cpu", torch_dtype=torch.float16).eval()
# model = AutoModelForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", device_map="cpu", torch_dtype=torch.float16).eval()
# try:
#     model = AutoModelForCausalLM.from_pretrained("Yukang/Llama-2-7b-longlora-100k-ft", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b-32k", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("alpindale/Mistral-7B-v0.2-hf", trust_remote_code=True, device_map="cpu", torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-6B-200K", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")
# try:
#     model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-base-7b", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16).eval()
# except Exception as e:
#         print(f"加载模型失败: {e}")




# model = AutoModelForCausalLM.from_pretrained("Yukang/Llama-2-7b-longlora-100k-ft", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b-32k", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=access_token)
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("alpindale/Mistral-7B-v0.2-hf", trust_remote_code=True, device_map="cpu", torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-6B-200K", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-base-7b", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

# model = AutoModelForCausalLM.from_pretrained("google/recurrentgemma-2b", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="cpu", trust_remote_code=True, torch_dtype=torch.float16, token=access_token).eval()

"""
HF_API_TOKEN="your_access_token" python download.py
    
"""