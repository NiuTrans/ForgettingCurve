# Forgetting Curve
A Reliable Method to Evaluate Memorization Capability for all Long Context Models

# 环境配置
```
git clone https://github.com/1azybug/ForgettingCurve.git
cd ForgettingCurve
conda create -n forget python=3.10 -y
conda activate forget
# [cuda12.1 用于 xformers安装] https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers
conda install pytorch==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 模型准备
(Option)可以参考download.py预先下载模型，有些模型需要HF的Token，如meta-llama/Meta-Llama-3-8B
```
HF_API_TOKEN="your_access_token" python download.py
```

# 绘制在huggingface上开源的语言模型的遗忘曲线
model_id为huggingface上的模型
--repeat_time 控制每个长度重复测试的次数
--granularity 控制等分点的数量
--granularity_type "linear"表示长度线性增长，"log"表示长度以指数增长
--test_max_length 表示测试的最大长度
--training_len 绘制一个参考线，一般为训练长度

下面的代码表示绘制meta-llama/Meta-Llama-3-8B在依赖长度16000以内（实际上最长的token序列达32000）的遗忘曲线。每16000/32长度进行测试，重复测试10次。
```
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
```
![Meta-Llama-3-8B](./Meta-Llama-3-8B.png "Meta-Llama-3-8B")

更多模型的遗忘曲线绘制可以参考test.sh

# 绘制自定义模型的遗忘曲线
```
from draw import evaluate
evaluate(model, tokenizer, texts_or_ids=test_tokens, config=config)
```
model:自定义的模型
tokenizer:模型对应的分词器，当texts_or_ids是形状为[S]的token序列张量时不需要传入
texts_or_ids：可以是分词后的token序列张量、str或List[str]

对于普通的Transformer模型来说不需要使用config。
而对于segment-level循环的Transformer可以在config里实现一个teacher_forcing_forward的函数以进行更快更省的推理，详见example.py。
teacher_forcing_forward的输入是model, prompt_ids，输出是predicted_token_ids。


