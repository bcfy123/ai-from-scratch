import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# 简化版数据加载器：不创建 PyTorch Dataset/DataLoader，而是在每个训练步骤中
# 直接从 train.bin 或 val.bin 随机取若干段连续 token。
data_dir = "data"

def get_batch(split):
    """从指定数据集构造一个监督学习 batch，返回输入 x 和正确答案 y。

    设 token 序列为 [10, 21, 7, 42, 3]，某个样本的起点为 0、
    block_size 为 4，则：

        x = [10, 21,  7, 42]  # 模型已看到的上下文
        y = [21,  7, 42,  3]  # 每个位置应预测的“下一个 token”

    因而 x 和 y 的内容相同，只是 y 整体向左错开一个位置。
    """
    # np.memmap 将 .bin 映射到内存：可像 NumPy 数组一样按下标读取，
    # 却不必一次性把整个数据集加载进 RAM。dtype 必须和 prepare.py
    # 保存时的 np.uint16 保持一致。
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # 为 batch 中的每个样本随机选择一个起点。范围不能超过
    # len(data) - block_size，否则 x 或其右移一位后的 y 会越过数据末尾。
    # ix 的形状是 (batch_size,)，例如 batch_size=64 时包含 64 个起点。
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # 对每个起点 i，切出 block_size 个连续 token 作为输入。
    # astype(np.int64) 是必要的：Embedding 和交叉熵损失都要求 token ID 为 int64。
    # stack 后 x 的形状为 (batch_size, block_size)，即 (B, T)。
    # stack里的dim作用，新增一个维度，这个维度是stack里的数量，在这个维度上，每一个维度对应 之前的一个 tensor
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    # 标签从 i+1 开始，因此 y[t] 始终是 x[t] 的下一个 token。
    # 这正是 GPT 的训练目标：“根据当前位置及左侧内容预测下一个 token”。
    # y 与 x 形状相同，都是 (batch_size, block_size)。
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # 固定（pin）CPU 内存后，可用 non_blocking=True 异步将数据传到 GPU，
        # 使数据传输与 GPU 计算有机会重叠，提高吞吐量。
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        # CPU 或 Apple MPS 情况下，普通地把数据移动到模型所在设备。
        x, y = x.to(device), y.to(device)

    # 训练循环随后调用 model(x, y)：模型据此计算预测和交叉熵损失。
    return x, y