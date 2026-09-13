vocab = ["apple", "banana", "cherry"]

word_to_index = {word:idx for idx, word in enumerate(vocab)}

def one_hot_encoding(word, vocab, word_to_index):
    encoding = [0]*len(vocab)

    idx = word_to_index.get(word, -1)

    if idx!=-1:
        encoding[idx] = 1
    return encoding

print("one hot embedding: ")
for word in vocab:
    print(f"'{word}': {one_hot_encoding(word, vocab, word_to_index)}")