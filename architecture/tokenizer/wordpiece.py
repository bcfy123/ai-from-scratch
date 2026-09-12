import math
from collections import defaultdict

# 1. 准备训练语料
corpus = [
    "low", "low", "low", "low", "low",  # 5次
    "lower", "lower",                  # 2次
    "newest", "newest", "newest",      # 3次
    "wider", "wider",                  # 2次
]

def get_word_freqs(corpus):
    freqs = defaultdict(int)
    for word in corpus:
        freqs[word] += 1
    return freqs

def init_vocab_and_splits(word_freqs):
    """初始化基础字符词表，并为单词加上 ## 前缀切分"""
    vocab = set()
    splits = {}
    for word, freq in word_freqs.items():
        # 首字符不加 ##，后续字符加 ##
        split = [word[0]] + [f"##{c}" for c in word[1:]]
        splits[word] = split
        vocab.update(split)
    return vocab, splits

def compute_pair_scores(splits, word_freqs):
    """计算所有相邻 Pair 的 WordPiece Score: P(AB) / (P(A) * P(B))"""
    token_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    total_tokens = 0

    # 1. 统计当前所有 Token 和 Pair 的出现频次
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split)):
            token_freqs[split[i]] += freq
            total_tokens += freq
            if i < len(split) - 1:
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq

    # 2. 计算互信息得分
    scores = {}
    for (p1, p2), pair_freq in pair_freqs.items():
        merged_token = p1 + p2[2:] if p2.startswith("##") else p1 + p2
        
        score = (pair_freq * total_tokens) / (token_freqs[p1] * token_freqs[p2])
        scores[(p1, p2)] = score

    return scores

def merge_pair(p1, p2, splits):
    """在 splits 中将指定 Pair 合并为新 Token"""
    new_splits = {}
    merged_token = p1 + p2[2:] if p2.startswith("##") else p1 + p2

    for word, split in splits.items():
        i = 0
        new_split = []
        while i < len(split):
            if i < len(split) - 1 and split[i] == p1 and split[i+1] == p2:
                new_split.append(merged_token)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split
    return new_splits, merged_token

def train_wordpiece(corpus, target_vocab_size):
    word_freqs = get_word_freqs(corpus)
    vocab, splits = init_vocab_and_splits(word_freqs)

    while len(vocab) < target_vocab_size:
        scores = compute_pair_scores(splits, word_freqs)
        if not scores:
            break
        
        # 找出 Score 最大的 Pair
        best_pair = max(scores, key=scores.get)
        splits, merged_token = merge_pair(best_pair[0], best_pair[1], splits)
        
        vocab.add(merged_token)
        print(f"Merge {best_pair} -> '{merged_token}' | Score: {scores[best_pair]:.4f}")

    return vocab

# 训练词表（目标大小 13）
print("========== Training WordPiece ==========")
vocab = train_wordpiece(corpus, target_vocab_size=13)
print(f"\nFinal Vocab ({len(vocab)}):", sorted(list(vocab)))

def tokenize_word(word, vocab):
    """WordPiece 经典的 MaxMatch 贪心切分算法"""
    output_tokens = []
    start = 0
    
    while start < len(word):
        end = len(word)
        cur_substr = None
        
        # 从最长子串开始往回试（贪心匹配）
        while start < end:
            substr = word[start:end]
            # 如果不是单词开头，需要加上 ## 前缀去词表查
            if start > 0:
                substr = "##" + substr
                
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
            
        # 如果连单个字符都没匹配上，说明包含未知字符，标为 [UNK]
        if cur_substr is None:
            return ["[UNK]"]
            
        output_tokens.append(cur_substr)
        start = end  # 移动指针
        
    return output_tokens

print("\n========== Tokenization Demo ==========")
test_words = ["lowest", "wider", "unknown"]

for word in test_words:
    tokens = tokenize_word(word, vocab)
    print(f"'{word}' -> {tokens}")