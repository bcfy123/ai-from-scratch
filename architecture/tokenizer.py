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