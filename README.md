# Speculative_Decoding
手搓投机解码流程

# Steps
1.创建新虚拟环境
```sh
Windows
python -m venv .venv
.\.venv\Scripts\activate
```
```sh
Linux
python -m venv .venv
source .venv/bin/activate
```

2.安装依赖
```sh
pip install -r requirements.txt
python -c "import transformers; import datasets; import yaml; print('all imports ok')"
```

3.修改 huggingface Cache 路径！！！
```sh
export HF_HOME=/data/xiangyouLiu/huggingface_cache
```

3.下载符合当前 GPU 配套 CUDA 版本的 torch
```sh
pip install torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu128
```


4.（可选）修改 configs/default.yaml 训练数据集相关配置


5.（可选）配置 Huggingface 镜像源
```sh
export HF_ENDPOINT=https://hf-mirror.com
```

5.运行投机解码
```sh
python cli.py generate --speculative-decoding \
--target-model-path xxx --draft-model-path xxx --draft-length 8 --max-new-tokens 512 \
--prompt xx

```
