import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    把 0~255 的 byte 映射成可打印的 Unicode 字符
    从而让 BPE 可以直接在字符串上工作
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 不安全的unicode 用别的字符进行映射
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n+=1
    # 放对应的字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    提取word中所有相邻的token加入到集合
    word是由 256字节 对应的 unicode 字符组成
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class gpt2bpetokenizer:
    def __init__(self, encoder, bpe_merges):
        self.encoder = encoder
        self.decoder = {
            v:k 
            for k, v in encoder.items()
        }
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {
            v: k
            for k, v in self.byte_encoder.items()
        }
        self.bpe_ranks = {
            merge: i
            for i, merge in enumerate(bpe_merges)
        }
        self.cache = {}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|   # 1. 匹配常见的英文缩写 (如 's, 're)
                ?\p{L}+|                 # 2. 匹配可能带有一个前导空格的“纯字母单词”
                ?\p{N}+|                 # 3. 匹配可能带有一个前导空格的“纯数字”
                ?[^\s\p{L}\p{N}]+|       # 4. 匹配可能带有一个前导空格的“标点符号/特殊字符”
                \s+(?!\S)|               # 5. 匹配末尾连续的空白符
                \s+""",                  # 6. 匹配其他空白符
            re.VERBOSE
        )
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(
                pairs,
                key = lambda pair:
                self.bpe_ranks.get(
                    pair,
                    float("inf")
                )
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word=[]
            i = 0
            while i < len(word):
                try:
                    # 找下一个 first
                    j = word.index(first, i)
                    # first 前面的内容直接保留
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 后面没有 first 了
                    new_word.extend(word[i:])
                    break
                # first + second 正好匹配
                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(
                        first + second
                    )
                    i += 2
                else:
                    new_word.append(
                        word[i]
                    )
                    i += 1
            # tuple 方便下一轮处理
            word = tuple(new_word)
            if len(word) == 1:
                break
            # 重新计算 pair
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_bytes = token.encode("utf-8")
            token = "".join(
                self.byte_encoder[b]
                for b in token_bytes
            )
            token_bpe = self.bpe(token)
            for bpe_token in token_bpe.split(" "):

                bpe_tokens.append(
                    self.encoder[bpe_token]
                )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join(
            self.decoder[token]
            for token in tokens
        )
        text = bytearray(
            self.byte_decoder[c]
            for c in text
        )
        return text.decode(
            "utf-8",
            errors="replace"
        )

def load_gpt2_tokenizer(
    encoder_path,
    vocab_bpe_path
):
    with open(
        encoder_path,
        "r",
        encoding="utf-8"
    ) as f:

        encoder = json.load(f)
    with open(
        vocab_bpe_path,
        "r",
        encoding="utf-8"
    ) as f:
        bpe_data = f.read()
    merges = [
        tuple(line.split())
        for line in bpe_data.split("\n")[1:-1]
    ]
    return gpt2bpetokenizer(
        encoder,
        merges
    )