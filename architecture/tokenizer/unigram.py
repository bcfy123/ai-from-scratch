import math

vocab = {
    "l": 0.10,
    "o": 0.10,
    "w": 0.10,
    "e": 0.10,
    "r": 0.10,

    "lo": 0.15,
    "ow": 0.05,
    "low": 0.25,
    "er": 0.15,

    "lowe": 0.10,
    "lower": 0.20,
}

def word_probability(word, vocab):
    n = len(word)

    # dp[i] = 前 i 个字符组成 word[:i] 的总概率
    dp = [0.0] * (n + 1)
    dp[0] = 1.0

    for i in range(n):
        if dp[i] == 0:
            continue

        for token, prob in vocab.items():
            if word.startswith(token, i):
                j = i + len(token)
                dp[j] += dp[i] * prob

    return dp[n]

def word_probability_pull(word, vocab):
    n = len(word)
    dp = [0.0] * (n + 1)
    dp[0] = 1.0  # 空字符概率为 1.0

    # 外层遍历终点 j (1 到 n)
    for j in range(1, n + 1):
        # 内层遍历起点 i (0 到 j-1)
        for i in range(j):
            # 切出从 i 到 j 的子串
            sub_str = word[i:j]

            # 如果这个子串在词表里，且起点 i 是可达的
            if sub_str in vocab and dp[i] > 0:
                prob = vocab[sub_str]
                # 把从 i 转移到 j 的概率加到 dp[j]
                dp[j] += dp[i] * prob

    return dp[n]

def word_loss(word, vocab):
    p = word_probability(word, vocab)
    if p==0:
        return float("inf")
    return -math.log(p)

def corpus_loss(corpus, vocab):
    total = 0.0

    for word in corpus:
        total += word_loss(word, vocab)

    return total

def best_segmentation(word, vocab):
    n = len(word)

    dp = [(-float("inf"), []) for _ in range(n + 1)]
    dp[0] = (0.0, [])

    for i in range(n):
        if dp[i][0] == -float("inf"):
            continue

        for token, prob in vocab.items():

            if word.startswith(token, i):
                j = i + len(token)

                score = dp[i][0] + math.log(prob)

                if score > dp[j][0]:
                    dp[j] = (
                        score,
                        dp[i][1] + [token]
                    )

    return dp[n]

corpus = [
    "low",
    "lower",
]

original_loss = corpus_loss(corpus, vocab)

results = []

for token in vocab:
    new_vocab = vocab.copy()
    del new_vocab[token]
    new_loss = corpus_loss(corpus, new_vocab)
    delta = new_loss - original_loss
    results.append((delta, token, new_loss))

results.sort()
delta, token_to_remove, new_loss = results[0]