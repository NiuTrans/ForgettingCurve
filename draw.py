import torch
import random
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='MemBench Configuration')

    # 添加参数及其默认值
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf", help='hf_repository')
    parser.add_argument('--text_path', type=str, default="emozilla/pg19-test", help='hf_datasets')
    parser.add_argument('--repeat_time', type=int, default=10, help='Number of times to repeat the experiment')
    parser.add_argument('--granularity', type=int, default=32, help='Granularity of the experiment')
    parser.add_argument('--granularity_type', type=str, choices=['linear', 'log'], default='linear', help='Type of granularity')
    parser.add_argument('--data_type', type=str, default='order', help='Type of data to use')
    parser.add_argument('--test_max_length', type=int, default=20000, help='Maximum length for testing')
    parser.add_argument('--training_len', type=int, default=None, help='Length of training data (None for default)')
    parser.add_argument('--title', type=str, default='MemBench', help='Title of the experiment')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    # 解析参数
    args = parser.parse_args()

    # 如果需要，可以将参数转换为字典
    config = vars(args)

    return config


def check_assumptions(data1, data2=None):
    """
    使用Shapiro-Wilk检验检查数据的正态性，并使用Levene检验检查方差齐性。
    
    参数:
    data1: 第一组数据。
    data2: 第二组数据。
    
    返回:
    一个字典，包含正态性和方差齐性的检验结果。
    """
    # 正态性检验
    shapiro_p_value1 = stats.shapiro(data1)[1]
    shapiro_p_value2 = stats.shapiro(data2)[1]
    normality = shapiro_p_value1 > 0.05 and shapiro_p_value2 > 0.05
    
    # 方差齐性检验
    levene_p_value = stats.levene(data1, data2)[1]
    variance_homogeneity = levene_p_value > 0.05
    
    # 结果
    assumptions_met = {
        'normality': bool(normality),
        'variance_homogeneity': bool(variance_homogeneity)
    }
    return assumptions_met


def generate_random_integer_list(min_value, max_value, length):
    return [random.randint(min_value, max_value) for _ in range(length)]

def get_random_example(config,length):
    min_value = 10
    max_value = 31999
    bos_token_id=1
    eos_token_id=2
    copy_tokens = generate_random_integer_list(min_value, max_value, length)
    lm_tokens = copy_tokens.copy()
    copy_tokens = [bos_token_id] + copy_tokens + [bos_token_id] + copy_tokens + [eos_token_id] 

    # no same tokens prepend, test the LM acc%
    irrelevant_tokens = generate_random_integer_list(min_value, max_value, length)
    if "irrelevant_text" in config:
        irrelevant_tokens = config["irrelevant_text"][-length:]
    lm_tokens = [bos_token_id] + irrelevant_tokens + [bos_token_id] + lm_tokens + [eos_token_id] 
    mask = [0 for _ in range(1+length+1)] + [1 for _ in range(length)] + [0]
    return lm_tokens, copy_tokens, mask

def get_order_example(config, length, tokens):

    max_len = tokens.size(0)
    begin_loc = random.randint(0, max_len-length*3)
    end_loc = begin_loc + length

    bos_token_id=1
    eos_token_id=2

    copy_tokens = tokens[begin_loc:end_loc].tolist()
    copy_tokens = [bos_token_id] + copy_tokens + [bos_token_id] + copy_tokens + [eos_token_id] 
    mask = [0 for _ in range(1+length+1)] + [1 for _ in range(length)] + [0]

    # no same tokens prepend, test the LM acc%
    lm_tokens = tokens[begin_loc:end_loc].tolist()

    irrelevant_tokens = tokens[-length:].tolist()
    if "irrelevant_text" in config:
        irrelevant_tokens = config["irrelevant_text"][-length:]
    lm_tokens = [bos_token_id] + irrelevant_tokens + [bos_token_id] + lm_tokens + [eos_token_id] 


    return lm_tokens, copy_tokens, mask

def mem_forward(model, prompt, memory):
    work_size = memory.work_size
    s = prompt.size(1)
    # print(f"[DEBUG] len={s}, prompt shape:{prompt.shape}")
    all_ids = []
    with torch.no_grad(): 
        for beg in range(0, s, work_size):
            forward_prompt = prompt[:,beg:beg+work_size] 
            # print(f"[DEBUG] prompt[:,{beg}:{beg+work_size}] ")
            output = model(forward_prompt,past_key_values=memory)
            logits = output.logits
            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_token_ids = predicted_token_ids.squeeze().tolist()
            all_ids += predicted_token_ids
    return all_ids
        
    

def get_acc(model, example, mask, length, config, memory=None):
    prompt = torch.tensor(example, dtype=torch.long)
    prompt = prompt[None,:].to(model.device)
    # [1,S]
    with torch.no_grad():
        
        if config["teacher_forcing_forward"] is None:
            if memory is None:
                output = model(prompt)
                logits = output.logits
                predicted_token_ids = torch.argmax(logits, dim=-1)
                predicted_token_ids = predicted_token_ids.squeeze().tolist()
            else:
                # clear the memory
                memory.reset_memory()
                predicted_token_ids = mem_forward(model, prompt, memory)
        else:
            predicted_token_ids = config["teacher_forcing_forward"](model, prompt)

    # shift right
    predicted_token_ids = [example[0]] + predicted_token_ids[:-1]

    mask = torch.tensor(mask, dtype=torch.bool)
    src = torch.tensor(example, dtype=torch.long)[mask]
    tgt = torch.tensor(predicted_token_ids, dtype=torch.long)[mask]

    # pre 50% tokens for few-shot prompt, only calculate the accuracy of the last 50% token
    result = (src[-length//2:] == tgt[-length//2:]).float().mean().item()
    return result


def get_length_result(config, model, length, tokens=None, test_times=100, data_type="random", memory=None):

    lm_results = []
    copy_results = []
    print("-"*80)
    print(data_type)
    print(f"length:{length}")
    for _ in tqdm(range(test_times)):
        if data_type=="random":
            lm_tokens, copy_tokens, mask = get_random_example(config, length)
        else:
            lm_tokens, copy_tokens, mask = get_order_example(config, length, tokens)
        
        lm_acc = get_acc(model, lm_tokens, mask, length, config, memory)
        copy_acc = get_acc(model, copy_tokens, mask, length, config, memory)

        lm_results.append(lm_acc)
        copy_results.append(copy_acc)
    
    
    check_result = check_assumptions(lm_results,copy_results)
    print(check_result)
    # assert check_result['normality']
    # assert check_result['variance_homogeneity']
    t_stat, p_value = stats.ttest_rel(lm_results, copy_results)
    
    # all zero is the same.
    if np.isnan(p_value):
        p_value = 1.0
    print(f"p_value:{p_value}")

    lm_mean = np.mean(lm_results)
    lm_std = np.std(lm_results)
    copy_mean = np.mean(copy_results)
    copy_std = np.std(copy_results)
    print(f"lm_mean:{lm_mean}, lm_std:{lm_std}")
    print(f"copy_mean:{copy_mean}, copy_std:{copy_std}")
    # print(results)
    return {
        "length":int(length), 
        "lm_mean":lm_mean, 
        "lm_std":lm_std,
        "copy_mean":copy_mean,
        "copy_std":copy_std,
        "lm_accs":lm_results,
        "copy_accs":copy_results,
        "p_value":p_value,
        "normality":check_result
    }

def get_result(config, model, tokens=None, test_times=100, data_type="random", memory=None):
    # if os.path.exists(config["title"]+".json"):
    #     with open(config["title"]+".json") as f:
    #         result = json.load(f)
    #     return result
    
    
    if config["granularity_type"] == "log":
        lens = np.unique(np.geomspace(8, config["test_max_length"], num=config["granularity"], dtype=int))
    elif config["granularity_type"] == "linear":
        lens = np.linspace(8, config["test_max_length"], num=config["granularity"], dtype=int)
    else:
        raise ValueError("config[\"granularity_type\"] must be \"log\" or \"linear\".")
    
    results = []
    for length in lens:
        try:
            result = get_length_result(config, model, length=length, tokens=tokens, test_times=test_times, data_type=data_type, memory=memory)
            results.append(result)
            # print(result)
        except Exception as e:
            print(f"len:{length} Error :{e}")
            break
    with open(config["title"]+".json", "w") as f:
        json.dump(results, f, indent=4)
    return results

def draw_work_memory(config, model, tokens=None, test_times=100, data_type="random", memory=None):

    results = get_result(config, model, tokens=tokens, test_times=test_times, data_type=data_type, memory=memory)
    length = [result['length'] for result in results]
    lm_acc = [result['lm_mean'] for result in results]
    p_value = [result['p_value'] for result in results]
    lm_acc_upper = [result['lm_mean']+result['lm_std'] for result in results]
    lm_acc_lower = [result['lm_mean']-result['lm_std'] for result in results]

    copy_acc = [result['copy_mean'] for result in results]
    copy_acc_upper = [result['copy_mean']+result['copy_std'] for result in results]
    copy_acc_lower = [result['copy_mean']-result['copy_std'] for result in results]

    plt.figure(figsize=(10, 5))

    
    if config["granularity_type"]=="log":
        plt.xlabel("Sequence Length(log-scale)")
        plt.semilogx(length, lm_acc, label='lm_accuracy', color='blue')
        plt.semilogx(length, copy_acc, label='copy_accuracy', color='orange')
    else:
        plt.xlabel("Sequence Length")
        plt.plot(length, lm_acc, label='lm_accuracy', color='blue')
        plt.plot(length, copy_acc, label='copy_accuracy', color='orange')

    plt.fill_between(length, lm_acc_lower, lm_acc_upper, color='blue', alpha=0.1)
    plt.fill_between(length, copy_acc_lower, copy_acc_upper, color='orange', alpha=0.1)
    
    if config["training_len"]:
        plt.axvline(x=config["training_len"], color='red', linestyle='--', label=f'support_len={config["training_len"]}')
        # plt.text(config["training_len"], 0.9, f'support_len={config["training_len"]}', verticalalignment='bottom', horizontalalignment='right')
        plt.axvline(x=config["training_len"]//2, color='green', linestyle='--', label=f'half_support_len={config["training_len"]//2}')
        # plt.text(config["training_len"]//2, 0.2, f'half_support_len={config["training_len"]//2}', verticalalignment='bottom', horizontalalignment='right')
    
    
    # plt.scatter(length, p_value, color='green', marker='o', label='p_value')
    # plt.axhline(y=0.05, color='grey', linestyle='--', label='p_value = 0.05')
    
    plt.ylabel("Accuracy")
    plt.title(config["title"])
    plt.legend()
    plt.grid(True)
    plt.show()
    save_png_path = os.path.join(config['save_path'],f'{config["title"]}.png')
    print(f"png save to {save_png_path}")
    plt.savefig(save_png_path)
    plt.close()
    
def prepare_ids(tokenizer, texts_or_ids):
    """

    Returns:
        LongTensor: shape:[S]
    """
    
    tokens = None
    
    # List[str]
    if isinstance(texts_or_ids, list):
        all_ids = []
        for example in tqdm(texts_or_ids, desc="Processing examples"):
            ids = tokenizer(example)["input_ids"]
            all_ids += ids
        tokens = torch.tensor(all_ids)
        return tokens

    elif isinstance(texts_or_ids, str):
        all_ids = tokenizer(texts_or_ids)["input_ids"]
        tokens = torch.tensor(all_ids)
        return tokens

    return texts_or_ids

def get_default_config(cfg):
    config = {
        "repeat_time":10,
        "granularity":32,
        "granularity_type":"linear",  # or log
        "data_type":"order",
        "test_max_length":20000,
        "training_len":None,
        "title":"MemBench",
        "save_path":".",
        "teacher_forcing_forward":None
    }
    config.update(cfg)
    return config

def evaluate(model, tokenizer, texts_or_ids, config=None):
    tokens = prepare_ids(tokenizer, texts_or_ids)
    config = get_default_config(config)
    
    # print(config)
    draw_work_memory(config, model, tokens=tokens, test_times=config["repeat_time"], data_type=config["data_type"])






if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    config = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    if config["device"] == 'cpu':
        # model = AutoModelForCausalLM.from_pretrained(config["model_id"], trust_remote_code=True, device_map=config["device"], use_mamba_kernels=False, torch_dtype=torch.float16).eval()
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], trust_remote_code=True, device_map=config["device"], torch_dtype=torch.float16).eval()
    else:
        # model = AutoModelForCausalLM.from_pretrained(config["model_id"], trust_remote_code=True, device_map="auto", use_mamba_kernels=False, torch_dtype=torch.float16).eval()
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], trust_remote_code=True, device_map="auto", torch_dtype=torch.float16).eval()
        
    
    
    if config["text_path"] == "emozilla/pg19-test":
        dataset = load_dataset(config["text_path"])
        text_list = []
        for example in tqdm(dataset['test'], desc="Processing examples"):
            text_list.append(example["short_book_title"]+"\n")
            text_list.append(example["text"]+"\n")
    
    evaluate(model, tokenizer, texts_or_ids=text_list, config=config)




"""
python ./draw.py --model_id meta-llama/Llama-2-7b-hf --title Llama-2-7b-hf
"""
