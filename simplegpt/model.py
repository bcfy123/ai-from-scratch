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
        # 没有词表大小，就无法创建“token ID -> 向量”以及“向量 -> token 分数”的层；
        # 没有 block_size，就无法创建位置向量表。
        assert config.vocab_size is not None
        assert config.block_size is not None

        # 保留这份图纸，forward()、生成代码和保存 checkpoint 时都会使用它。
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
            # h 代表 hidden blocks：按顺序堆叠 n_layer 个相同结构、不同参数的
            # Transformer Block。ModuleList 可确保这些 Block 的参数被 PyTorch 注册。
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 所有 Block 结束后的最终 LayerNorm，稳定送往输出层的向量分布。
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

        # 仅打印模型规模，便于确认当前配置实际创建了多大的网络。
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
