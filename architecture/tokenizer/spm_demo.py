import sentencepiece as spm

# 1. 准备训练语料文件
corpus_text = """
low lower lowest
new newer newest
wide wider widest
unaffordable
"""

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus_text.strip())

# 2. 训练 BPE 模型
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="bpe_model",
    vocab_size=20,             # 词表大小
    model_type="bpe",          # 选 BPE 算法
    character_coverage=1.0,
    user_defined_symbols=[]
)

# 3. 训练 Unigram 模型 (SentencePiece 默认算法)
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="unigram_model",
    vocab_size=20,
    model_type="unigram"       # 选 Unigram 算法
)

print("========== SentencePiece (BPE) ==========")
sp_bpe = spm.SentencePieceProcessor(model_file="bpe_model.model")
text = "lowering unaffordable"
print("Tokens:", sp_bpe.encode_as_pieces(text)) # 字符串列表
print("IDs:   ", sp_bpe.encode_as_ids(text))    # 整数 ID 列表

print("\n========== SentencePiece (Unigram) ==========")
sp_unigram = spm.SentencePieceProcessor(model_file="unigram_model.model")
print("Tokens:", sp_unigram.encode_as_pieces(text))
print("IDs:   ", sp_unigram.encode_as_ids(text))