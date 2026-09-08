from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.vocab={}
        self.merges=[]

    def build_vocab(self, corpus):
        """
        初始化vocabulary
        将文本拆成字符，并统计每个单词出现次数
        """
        vocab = Counter()

        for word in corpus:
            # </w>特殊终结符，代表词尾
            tokens = tuple(list(word)+["</w>"])
            # 字典的key必须是不可变对象，如tuple
            vocab[tokens]+=1
        return vocab

    def get_stats(self, vocab):
        """
        统计相邻token pair的频率
        """
        pairs = Counter()

        for tokens, freq in vocab.items():
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair]+=freq
        return pairs

    def merge_pair(self, vocab, pair):
        """
        根据pair进行一次merge
        """
        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i=0
            while i<len(tokens):
                if(i<len(tokens)-1 
                   and tokens[i]==pair[0] 
                   and tokens[i+1]==pair[1]):
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            # 防止合并后 出现相同的 词
            new_vocab[tuple(new_tokens)]+=freq
        return new_vocab

    def train(self, corpus, num_merges=10):
        vocab = self.build_vocab(corpus)
        for i in range(num_merges):
            # 每个词计算相邻token的频率
            stats = self.get_stats(vocab)
            if not stats:
                break
            # 取最高频率的pair
            best_pair, frequency = stats.most_common(1)[0]

            self.merges.append(best_pair)
            # 进行一次merge
            vocab = self.merge_pair(vocab, best_pair)

        tokens = set()
        # 把每个词的token都提取出来，装进大集合
        for word_tokens in vocab:
            tokens.update(word_tokens)

        # 按照字典序排序，将下标索引作为token id
        tokens.add("<UNK>")
        self.vocab = {
            token: idx
            for idx, token in enumerate(sorted(tokens))
        }

    def encode_word(self, word):
        """
        对于新的word，进行token拆分和合并
        """
        tokens = list(word) + ["</w>"]
        # 按照训练阶段产生的 merge 顺序执行
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        """
        bpe train后得到 self.vocab
        基于self.vocab 可针对一段文本进行tokenize
        """
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(
                token,
                self.vocab["<UNK>"]
            ))
        return tokens, ids

    def decode(self, ids):
        id_to_token = {
            idx:token
            for token, idx in self.vocab.items()
        }
        tokens = [
            id_to_token.get(i, "<UNK>")
            for i in ids
        ]
        text = ""
        for token in tokens:
            if token=="</w>":
                text+=" "
            elif token.endswith("</w>"):
                text+=token[:-4]
                text+=" "
            else:
                text+=token
        return text.strip()

    def print_vocab(self, vocab):
        for tokens, freq in vocab.items():
            print(
                f"{freq:2d}:{' '.join(tokens)}"
            )

if __name__ == "__main__":

    corpus = [
        "low",
        "lower",
        "lowest",
        "newest",
        "widest",
    ]

    tokenizer = BPETokenizer()

    tokenizer.train(
        corpus,
        num_merges=10
    )

    print("\n==============================")
    print("Final vocabulary")
    print("==============================")

    for token, idx in tokenizer.vocab.items():
        print(f"{idx:3d} : {token}")

    print("\n==============================")
    print("Merge rules")
    print("==============================")

    for i, merge in enumerate(tokenizer.merges):
        print(
            f"{i + 1:2d}. "
            f"{merge[0]} + {merge[1]}"
            f" -> {merge[0] + merge[1]}"
        )

    print("\n==============================")
    print("Encoding")
    print("==============================")

    text = "lowest price you can undertake"

    tokens, ids = tokenizer.encode(text)

    print("text   :", text)
    print("tokens :", tokens)
    print("ids    :", ids)

    print("\n==============================")
    print("Decoding")
    print("==============================")

    print(
        "decoded:",
        tokenizer.decode(ids)
    )