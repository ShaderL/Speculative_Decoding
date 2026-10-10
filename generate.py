import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from model_miniLLaMa import ModelConfig, MiniLLaMA

def load_model(
    checkpoint_path,
    device,
):
    """
    从 checkpoint 中恢复 MiniLLaMA。

    checkpoint 中保存了：
        - model_state_dict
        - model_config
        - training_config
        - optimizer_state_dict
        - epoch
        - loss
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_config_dict = checkpoint["model_config"]

    model_config = ModelConfig(
        **model_config_dict
    )

    model = MiniLLaMA(
        model_config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print(
        f"Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint loss: "
        f"{checkpoint.get('loss', 'unknown')}"
    )

    return model, model_config


def apply_top_k(
    logits,
    top_k,
):
    """
    只保留概率最高的 top_k 个 token。

    logits:
        [B, V]

    其余 token 的 logit 设置为 -inf。
    """

    if top_k <= 0:
        return logits

    vocab_size = logits.size(-1)

    top_k = min(
        top_k,
        vocab_size,
    )

    values, _ = torch.topk(
        logits,
        top_k,
        dim=-1,
    )

    # 第 k 大的 logit
    min_values = values[:, -1].unsqueeze(-1)

    logits = logits.masked_fill(
        logits < min_values,
        float("-inf"),
    )

    return logits


def apply_top_p(
    logits,
    top_p,
):
    """
    Nucleus Sampling。

    保留累计概率达到 top_p 所需的最小 token 集合。

    例如：

        top_p = 0.9

    就只保留累计概率约 90% 的 token。
    """

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    sorted_probs = F.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    sorted_indices_to_remove = (
        cumulative_probs > top_p
    )

    # 保留第一个超过 top_p 的 token
    sorted_indices_to_remove[:, 1:] = (
        sorted_indices_to_remove[:, :-1].clone()
    )

    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = torch.zeros_like(
        logits,
        dtype=torch.bool,
    )

    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    logits = logits.masked_fill(
        indices_to_remove,
        float("-inf"),
    )

    return logits


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    do_sample=True,
):
    """
    根据最后一个位置的 logits 选择下一个 token。

    logits:
        [B, V]

    返回：
        next_token:
        [B, 1]
    """


    if not do_sample:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )


    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0"
        )

    logits = logits / temperature

    logits = apply_top_k(
        logits,
        top_k,
    )

    logits = apply_top_p(
        logits,
        top_p,
    )

    probs = F.softmax(
        logits,
        dim=-1,
    )

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    """
    根据 prompt 生成文本。

    参数：

        prompt:
            输入文本。

        max_new_tokens:
            最多生成多少个新 token。

        do_sample:
            False -> Greedy
            True  -> Sampling

        temperature:
            控制随机程度。

        top_k:
            只从概率最高的 K 个 token 中采样。
            0 表示关闭。

        top_p:
            Nucleus Sampling。
            1.0 表示关闭。
    """

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
    )

    input_ids = input_ids.to(device)

    for _ in range(max_new_tokens):

        # 如果生成长度超过模型 block_size，
        # 只保留最后 block_size 个 token。
        input_context = input_ids[
            :, -model.config.block_size:
        ]

        logits = model(
            input_context
        )

        next_token_logits = logits[
            :, -1, :
        ]

        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    generated_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )

    return generated_text


def generate_from_checkpoint(
    checkpoint_path,
    prompt,
    device=None,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    tokenizer_name="gpt2",
):
    """
    从 checkpoint 直接生成文本。

    这是给 cli.py 调用的主要接口。
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    tokenizer = GPT2TokenizerFast.from_pretrained(
        tokenizer_name
    )


    model, model_config = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return text


if __name__ == "__main__":
    raise RuntimeError(
        "Please run generation through cli.py."
    )
