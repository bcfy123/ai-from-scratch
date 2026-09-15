import os
import pickle
import requests
import numpy as np

# __file__代表当前文件所在的路径
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

# 若尚未下载原始语料，就从 Karpathy 的 char-rnn 仓库下载它。
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# set(data) 去重得到语料中出现过的所有字符；sorted 保证字符表的顺序稳定。
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# 建立双向“词表”（这里严格说是“字符表”）
# stoi: string to integer，例如 stoi['A'] -> 18
# itos: integer to string，例如 itos[18] -> 'A'
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    """把字符串编码为 token ID 列表，供模型输入使用。"""
    return [stoi[c] for c in s]

def decode(l):
    """把模型输出的 token ID 列表还原为字符串。"""
    return ''.join([itos[i] for i in l])

# 按文本原来的顺序切分：前 90% 训练，后 10% 验证。
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
# 模型不处理字符串，先将两部分文本全部编码为token ID。
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 使用 uint16（无符号 16 位整数）紧凑保存。词表只有 65 个字符，
# 所以每个 ID 都远小于 uint16 可表示的最大值 65535。
# train.py 会通过 numpy.memmap 读取这些二进制文件，而无需一次性加载全部数据。
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 保存词表元数据。sample.py 会读取它，将输入提示词编码，并把生成结果解码。
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)