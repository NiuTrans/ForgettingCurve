from draw import evaluate
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import random
import torch
import argparse
import json
from transformers import StaticCache


def parse_args():
    parser = argparse.ArgumentParser(description='MemBench Configuration')

    # 添加参数及其默认值
    parser.add_argument('--model_path', type=str, help='State dict file of your model')
    parser.add_argument('--model_config', type=str, help='Config file of your model')
    parser.add_argument('--repeat_time', type=int, default=10, help='Number of times to repeat the experiment')
    parser.add_argument('--granularity', type=int, default=32, help='Granularity of the experiment')
    parser.add_argument('--granularity_type', type=str, choices=['linear', 'log'], default='linear', help='Type of granularity')
    parser.add_argument('--data_type', type=str, default='order', help='Type of data to use')
    parser.add_argument('--test_max_length', type=int, default=20000, help='Maximum length for testing')
    parser.add_argument('--training_len', type=int, default=None, help='Length of training data (None for default)')
    parser.add_argument('--title', type=str, default='MemBench', help='Title of the experiment')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save results')

    # 解析参数
    args = parser.parse_args()

    # 如果需要，可以将参数转换为字典
    config = vars(args)

    return config

def llama_example():
    random.seed(0)
    torch.manual_seed(0)
    config = parse_args()
    # Load your model
    with open(config["model_config"]) as f:
            cfg=json.load(f)
    cfg = LlamaConfig(**cfg["model_config"])
    print(cfg)
    model = LlamaForCausalLM(cfg)
    state_dict = torch.load(config["model_path"])
    model.load_state_dict(state_dict)
    model.to('cuda')
    
    # Load your Tokens tokenized by your tokenizer
    # Tokens' shape: [S], S is the sequence_len
    tokens_path = 'test.pt'
    test_tokens = torch.load(tokens_path)
    
    # if your teacher forcing forward is special, like Transformer-xl，Mistral
    # you need to code your teacher forcing forward
    # which accept your model and prompt_ids，the prmopt_ids' shape is [batch_size,seq_len]， batch_size is 1
    def teacher_forcing_forward(model, prompt_ids):
        output = model(prompt_ids)
        logits = output.logits
        predicted_token_ids = torch.argmax(logits, dim=-1)
        predicted_token_ids = predicted_token_ids.squeeze().tolist()
        # return the list of next token which model predict 
        return predicted_token_ids
    
    config["teacher_forcing_forward"] = teacher_forcing_forward
    
    # if texts_or_ids is tensor,  we can pass None to tokenizer
    evaluate(model, tokenizer=None, texts_or_ids=test_tokens, config=config)

if __name__ == "__main__":
    llama_example()
