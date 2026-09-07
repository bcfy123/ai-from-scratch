from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.vocab={}
        self.merges=[]

    def build_vocab(self, corpus):
        """
        初始化vocabulary
        将文本拆成字符，并统计每个字符出现次数
        """
        vocab = Counter()

        for word in corpus:
            # </w>特殊终结符，代表词尾
            tokens = tuple(list(word)+["</w>"])
            # 字典的key必须是不可变对象，如tuple
            vocab[tokens]+=1
        return vocab