import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_miniLLaMa import ModelConfig, MiniLLaMA as Transformer
from transformers import GPT2TokenizerFast


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]

    config = ModelConfig(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        d_ffn=model_config["d_ffn"],
        num_heads=model_config["num_heads"],
        rms_norm_eps=model_config["rms_norm_eps"],
    )

    model = Transformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def sample(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def speculative_generate(
        target_model_path,
        draft_model_path,
        draft_length,
        prompt,
        device,
        max_new_tokens,
        tokenizer_name):

    target_model = load_model(target_model_path, device)
    draft_model = load_model(draft_model_path, device)

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    # input_ids [1,L]  # [1,L,dmodel] 是在 Embedding 后才变成的
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # L 代表现有文字长度，等于 prompt 长度 + 草稿长度
    L = input_ids.shape[1]
    prompt_len = L

    while input_ids.shape[1] - prompt_len < max_new_tokens:
        # draft_ids 纯草稿内容，不包含 prompt，不输入 model，所以采用 list 比较快
        draft_ids = []

        draft_logits = None

        # d_i_ids，负责掌管每轮输入进草稿模型的内容，即所有上下文, 注意，clone 的 input_ids 是 Tensor， 所以这里也是 Tensor
        # 需要与模型交互，使用 Tensor
        draft_input_ids = input_ids.clone()  # prompt，此后是 input_ids + new，每次都加上整段的 draft

        for _ in range(draft_length):
            draft_logits = draft_model(draft_input_ids)
            new_token_id = sample(draft_logits[:, -1, :])
            draft_ids.append(new_token_id.item())
            draft_input_ids = torch.cat([draft_input_ids, new_token_id], dim=1)  # tensor 使用 cat
            if (new_token_id.item() == tokenizer.eos_token_id):
                break

            # 此时 input_ids 比 draft_logits 多一行，因为最后一个 token id 拼接后没有再走一次 draft_model
            # 但是！draft_logits 最后一行已经是 最后一个 token 的分数了，而 target_logits 的最后一行是对下一个 token 的分数
            # 而且, 第一轮输出未采样 token 时，logits 的最后一行就是对第一个 token 的分数了
            # 同时，由于 Masked 自注意力机制的存在，导致每个 draft token 位置的上下文不论哪次输入都是一样的，因为只能看到前面的
            # 所以上下文一致，则 logits 必定一致。所以 draft logits 最后一版就可以保留前面每个 token 的 logits

        # 目标模型进行验证（draft_input_ids 是所有上下文）
        target_logits = target_model(draft_input_ids)
        accept_length = 0
        accept_ids = []

        # 考虑 EOS 后的实际草稿长度
        actual_len = len(draft_ids)

        for i in range(actual_len):
            # softmax 取概率分布，然后再取概率 p(x),q(x)
            # L 必须更新，因为草稿会越来越长
            p_probs = target_logits[0, L + i - 1].softmax(dim=-1)
            q_probs = draft_logits[0, L + i - 1].softmax(dim=-1)
            px = p_probs[draft_ids[i]]
            qx = q_probs[draft_ids[i]]

            # ids 都是 List ，没有 Batch 维度
            if px >= qx:
                # 直接接受
                accept_length += 1
                accept_ids.append(draft_ids[i])
                if draft_ids[i] == tokenizer.eos_token_id:
                    break
            else:
                r = random.uniform(0, 1)
                # 以 px/qx 概率接受
                if r <= px / qx:
                    accept_length += 1
                    accept_ids.append(draft_ids[i])
                else:
                    # 重新采样 norm(max(0,pn+1(x)−qn+1(x)))
                    # 逐元素减，然后取一下大于 0 ，再 norm 以下
                    adjusted = torch.clamp(p_probs - q_probs, min=0)
                    # 不可能 px 完全 < qx，只可能 px == qx, 而这种情况会在上面的 if 分支里，即直接接受
                    # 所以不可能除 0
                    adjusted = adjusted / adjusted.sum()
                    # 这里不能 accept_length += 1，因为补刀不计算进接受草稿长度
                    accept_ids.append(torch.multinomial(adjusted, 1).item())
                    break

        # 全部接受了，也要额外加上目标模型的最后一个采样
        # 为什么采用 draft_length ，因为如果 eos 中断的话不走这个“赠礼”分支
        if accept_length == draft_length:
            # 但是万一 eos 恰好是最后一个，且接受了，那也不加赠礼，继续后续流程
            if accept_ids[-1] != tokenizer.eos_token_id:
                extra_token = sample(target_logits[:, -1, :])
                accept_ids.append(extra_token.item())

        # 无论接受还是拒绝，直接拼接，因为拒绝的话剩下的就都是接受了的（所以 eos 也会被拼接进来）
        input_ids = torch.cat([input_ids, torch.tensor([accept_ids], dtype=torch.long, device=device)], dim=1)

        if input_ids[0, -1] == tokenizer.eos_token_id:
            break

        L = input_ids.shape[1]

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output
