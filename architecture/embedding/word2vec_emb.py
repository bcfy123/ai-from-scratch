import torch.nn as nn
import torch

class CBOW(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        embeddings = self.input_to_hidden(X)
        hidden_layer = torch.mean(embeddings, dim=0)
        output_layer = self.hidden_to_output(hidden_layer.unsqueeze(0))
        return output_layer

voc_size = 100
embedding_size = 2
cbow_model = CBOW(voc_size, embedding_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cbow_model.parameters(), lr=0.01)
context_words = [10, 20]
target_word = 30
# 转成 one-hot
X = torch.zeros(len(context_words), voc_size)
for i, word_id in enumerate(context_words):
    X[i, word_id] = 1
target = torch.tensor([target_word])

for epoch in range(10):
    # forward
    output = cbow_model(X)
    # 计算 loss
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f"epoch: {epoch}, loss: {loss.item()}")