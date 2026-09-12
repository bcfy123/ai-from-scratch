from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

# 1. 准备训练语料文件
corpus_text = """
low lower lowest
new newer newest
wide wider widest
unaffordable
"""

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus_text.strip())

# ① 初始化 BPE 模型
tokenizer_bpe = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer_bpe.pre_tokenizer = Whitespace() # 按空格粗切分

# ② 配置 Trainer (BPE 采用频率合并策略)
trainer_bpe = BpeTrainer(
    vocab_size=50,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
)

# ③ 训练模型
tokenizer_bpe.train(files=["corpus.txt"], trainer=trainer_bpe)

# ④ 推理分词
output = tokenizer_bpe.encode("lowering unaffordable")
print("========== HF Tokenizers (BPE) ==========")
print("Tokens:", output.tokens)
print("IDs:   ", output.ids)

# ① 初始化 WordPiece 模型
tokenizer_wp = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer_wp.pre_tokenizer = Whitespace()

# ② 配置 Trainer (使用基于似然 Score 的合并，指定 ## 前缀)
trainer_wp = WordPieceTrainer(
    vocab_size=50,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
    continuing_subword_prefix="##"
)

# ③ 训练模型
tokenizer_wp.train(files=["corpus.txt"], trainer=trainer_wp)

# ④ 推理分词（采用 MaxMatch 最长贪心匹配算法）
output = tokenizer_wp.encode("lowering unaffordable")
print("\n========== HF Tokenizers (WordPiece) ==========")
print("Tokens:", output.tokens)
print("IDs:   ", output.ids)

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace

# ① 初始化 Unigram 模型
tokenizer_unigram = Tokenizer(Unigram())
tokenizer_unigram.pre_tokenizer = Metaspace() # 使用 _ 替代空格

# ② 配置 Trainer (采用 EM 算法与 ΔLoss 剪枝)
trainer_unigram = UnigramTrainer(
    vocab_size=50,
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>"
)

# ③ 训练模型
tokenizer_unigram.train(files=["corpus.txt"], trainer=trainer_unigram)

# ④ 推理分词（采用 Viterbi 算法找最大 Log 概率路径）
output = tokenizer_unigram.encode("lowering unaffordable")
print("\n========== HF Tokenizers (Unigram) ==========")
print("Tokens:", output.tokens)
print("IDs:   ", output.ids)