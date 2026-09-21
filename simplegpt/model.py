import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd%config.n_head==0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
            q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
            v = v.view(B, T, self.n_head, C//self.n_haed).transpose(1,2)

            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att@v
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            y = self.resid_dropout(self.c_proj(y))
            return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    # 一条样本最多包含多少个 token，也就是模型可使用的最大上下文长度 T。
    block_size: int = 1024
    # token 的种类数 V。GPT-2 原词表有 50257 个 token；补到 50304（64 的倍数）
    # 可让某些硬件上的计算更高效。字符级 Shakespeare 会将它替换为 65。
    vocab_size: int = 50304
    # Transformer Block 的堆叠层数。每层都会做一次注意力和一次 MLP 变换。
    n_layer: int = 12
    # 每层注意力头的数量。n_embd 必须能被 n_head 整除。
    n_head: int = 12
    # 每个 token 在模型内部的向量维度 C，也称 embedding/channel dimension。
    n_embd: int = 768
    # 训练时随机置零一部分激活值的概率，用来缓解过拟合；推理时自动关闭。
    dropout: float = 0.0
    # Linear 和 LayerNorm 是否包含 bias 参数。False 略快、参数略少。
    bias: bool = True
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # ModuleDict 把各子模块以带名字的方式注册到 PyTorch；因此它们的参数会被
        # model.parameters()、优化器和 model.to(device) 自动发现和处理。
        self.transformer = nn.ModuleDict(dict(
            # word/token embedding table：形状 (V, C)。输入 token ID 后查表，
            # 将 (B, T) 的整数 idx 变成 (B, T, C) 的连续向量。
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding table：形状 (T_max, C)。相同字符在不同位置应有
            # 不同表示；例如句首的 'a' 与句末的 'a' 会加上不同的位置向量。
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # dropout 放在 embedding 之后；只在 model.train() 时生效。
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # language-model head：把每个位置的 C 维隐藏向量投影成 V 个 logit。
        # 输出形状为 (B, T, V)；每个 logit 是某个候选“下一个 token”的未归一化分数。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享（weight tying）：输入查表 wte 和输出层 lm_head 使用同一份矩阵参数，
        # 而非两份独立矩阵。这样既减少参数，也常能改善语言模型效果。
        # 两者形状都等价于 (V, C)：lm_head 的线性层在计算时使用其转置。
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # 初始化所有 Linear / Embedding 权重；训练会从这些随机初始值开始优化。
        self.apply(self._init_weights)

        # 对每个 Block 中残差分支的输出投影 c_proj 使用更小的初始标准差。
        # 层数越多，每条残差分支的初始影响越应小一些，避免信号在层间累积过大。
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters)
        if non_embedding:
            n_params = self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """执行一次 GPT 前向计算。
        参数：
            idx:     输入 token ID，形状为 (B, T)，例如 get_batch() 返回的 x。
            targets:  正确的下一个 token，形状也为 (B, T)，例如 get_batch() 返回的 y。
                      训练时传入它以计算损失；生成/推理时传 None。
        返回：
            logits: 每个候选 token 的未归一化分数。
                    训练时形状 (B, T, V)，推理时仅保留最后位置，为 (B, 1, V)。
            loss:   targets 存在时的交叉熵标量；推理时为 None。
        符号：B=batch size，T=当前序列长度，C=n_embd，V=vocab_size。
        """
        # idx 已由 get_batch() 移到模型所在设备；位置向量也必须创建在同一设备上。
        device = idx.device
        b, t = idx.size()

        # 当前输入不能超过模型设计时的位置向量表长度（block_size）。
        # 在字符级实验中，t 通常就是 256。
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 为序列中的每个位置生成编号 [0, 1, ..., T-1]，形状 (T,)。
        # 每个 batch 样本都复用相同的位置编号，因为它们都是独立的长度 T 片段。
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. token embedding：查表，把整数 token ID 从 (B, T) 转为 C 维向量，
        #    得到 tok_emb，形状 (B, T, C)。
        tok_emb = self.transformer.wte(idx)

        # 2. position embedding：查表，得到每个位置的 C 维向量，形状 (T, C)。
        #    它会在加法中自动广播到 batch 维度，变成每个样本各有一份的位置向量。
        pos_emb = self.transformer.wpe(pos)

        # 3. 对应位置的 token 向量 + 位置向量，让模型同时知道“是什么”和“在哪里”；
        #    dropout 只在训练模式启用。x 的形状仍为 (B, T, C)。
        x = self.transformer.drop(tok_emb + pos_emb)

        # 4. 依次经过 n_layer 个 Transformer Block。Block 内部包含
        #    因果自注意力、MLP 和残差连接；形状始终保持 (B, T, C)。
        for block in self.transformer.h:
            x = block(x)

        # 5. 最终 LayerNorm 后得到用于预测的隐藏状态，形状仍为 (B, T, C)。
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练路径：每个位置都要预测“下一个 token”，因此保留 T 个位置。
            # lm_head 将 C 维隐藏状态映射为 V 个候选 token 的分数，得到 (B, T, V)。
            logits = self.lm_head(x)

            # cross_entropy 需要类别维度位于最后，并将其余维度视为独立样本。
            # 因而把 (B, T, V) 压平为 (B*T, V)，并把 targets 的 (B, T)
            # 压平为 (B*T)。第 k 行 logits 与第 k 个 target 一一对应。
            # ignore_index=-1 允许未来在某些位置填 -1 来跳过该位置的损失。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理/生成路径：我们只需要用最后一个位置的隐藏状态预测“下一个 token”。
            # x[:, [-1], :] 中的 [-1]（列表）保留长度为 1 的时间维，形状为 (B, 1, C)，
            # 所以 logits 为 (B, 1, V)，而非 (B, V)。这样可少做前面 T-1 个位置的输出投影。
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # train.py 会用 loss.backward() 反向传播；sample.py 会根据 logits 抽样下一个 token。
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """以自回归方式续写 token 序列。

        参数：
            idx:            提示词的 token ID，形状 (B, T_prompt)。每轮生成后会不断变长。
            max_new_tokens: 要额外生成多少个 token。
            temperature:    控制随机性，必须大于 0；1.0 保持原分布，较小值更保守，
                            较大值更随机。
            top_k:          若非 None，仅允许概率最高的 k 个 token 被采样。

        返回：包含原提示词和新生成 token 的完整序列，形状
        (B, T_prompt + max_new_tokens)。

        注意：@torch.no_grad() 关闭梯度记录，避免生成时构建反向传播图、浪费显存。
        调用方还应先执行 model.eval()，使 Dropout 在生成时关闭。
        """

        # 每次循环只生成一个 token。新 token 会作为下一轮的上下文一部分，
        # 这就是 GPT 的自回归（autoregressive）生成。
        for _ in range(max_new_tokens):

            # 模型最多只能接收 block_size 个 token；若当前完整输出过长，
            # 仅把最后 block_size 个 token 送入模型。idx 本身仍保留完整结果，
            # 因此返回值不会丢失较早的生成内容，只是模型无法再关注它们。
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # forward(targets=None) 的推理路径只返回最后一个位置的 logits，形状 (B, 1, V)。
            # loss 为 None，因此用 '_' 忽略第二个返回值。
            logits, _ = self(idx_cond)

            # 去掉长度为 1 的时间维，得到 (B, V)：每个样本对“下一个 token”的全部候选分数。
            # temperature < 1 会拉大分数差距，使最高分 token 更容易被选中；
            # temperature > 1 会缩小差距，使分布更平坦、结果更随机。
            logits = logits[:, -1, :] / temperature

            # top-k sampling：可选地只保留每行分数最高的 k 个 token。
            # 其余 token 的 logit 设为 -∞，softmax 后概率会严格变成 0。
            if top_k is not None:
                # v 的形状为 (B, k)，每行是从大到小排列的前 k 个 logit。
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[:, [-1]] 是第 k 高的分数，形状 (B, 1)，可广播到 (B, V)。
                # 小于该阈值的全部候选都会被排除；使用 [-1] 保留二维形状以便广播。
                logits[logits < v[:, [-1]]] = -float('Inf')

            # softmax 把未归一化分数转成概率分布，形状仍为 (B, V)，每行之和为 1。
            probs = F.softmax(logits, dim=-1)

            # 按概率随机抽取一个 ID（不是总选最大值），每个样本取 1 个，因此形状为 (B, 1)。
            # 若想完全贪心生成，可改用 torch.argmax(probs, dim=-1, keepdim=True)。
            idx_next = torch.multinomial(probs, num_samples=1)

            # 沿时间维 dim=1 把新 token 接到序列末尾：(B, T) + (B, 1) -> (B, T+1)。
            # 下一轮循环会将更新后的 idx 再作为模型的输入上下文。
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu