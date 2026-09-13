import torch
import torch.nn as nn
import torch.optim as optim


voc_size = 100
embedding_size = 10

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Skip_gram(nn.Module):
    def __init__(self):
        super(Skip_gram, self).__init__()

        self.W = nn.Parameter(
            torch.randn(voc_size, embedding_size, dtype=dtype)
        )

        self.V = nn.Parameter(
            torch.randn(embedding_size, voc_size, dtype=dtype)
        )

    def forward(self, X):
        hidden_layer = torch.matmul(X, self.W)
        output_layer = torch.matmul(hidden_layer, self.V)

        return output_layer


model = Skip_gram().to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)