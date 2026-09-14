import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads ==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # view方法-改变形状，transpose方法-交换维度
        Q = Q.view(B,-1,self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B,-1,self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B,-1,self.num_heads, self.head_dim).transpose(1,2)
        # @-矩阵乘法，K最后2维进行交换
        scores = Q@K.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = attention@V
        # contiguous是真正修改物理内存顺序
        output = output.transpose(1,2).contiguous()
        output = output.view(B,-1, self.d_model)

        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(
            x,
            x,
            x,
            src_mask
        )
        x = self.norm1(x+self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x+self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )
        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )
        self.ffn = FeedForward(
            d_model,
            d_ff,
            dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        attn_output = self.self_attn(
            x,
            x,
            x,
            tgt_mask
        )
        x = self.norm1(
            x+self.dropout(attn_output)
        )
        attn_output = self.cross_attn(
            x,
            encoder_output,
            encoder_output,
            src_mask
        )
        x = self.norm2(x+self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm3(x+self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model,
                 num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(
            0, max_seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, d_model, 2,
                dtype=torch.float
            ) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.token_embedding(x)

        x = x + self.pe[:, :x.size(1)]

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                num_heads,
                d_ff,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):

        B, T = x.shape

        positions = torch.arange(
            T,
            device=x.device
        ).unsqueeze(0)

        x = (
            self.token_embedding(x)
            +
            self.position_embedding(positions)
        )

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                tgt_mask,
                src_mask
            )

        return x

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_seq_len,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):

        # Encoder
        encoder_output = self.encoder(
            src,
            src_mask
        )

        # Decoder
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask,
            src_mask
        )

        logits = self.output_projection(
            decoder_output
        )

        return logits

def causal_mask(size, device):
    mask = torch.tril(
        torch.ones(
            size,
            size,
            device=device
        )
    )
    return mask.unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":

    batch_size = 2

    src_len = 10
    tgt_len = 12

    src_vocab_size = 10000
    tgt_vocab_size = 10000

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_seq_len=128,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6
    )

    src = torch.randint(
        0,
        src_vocab_size,
        (batch_size, src_len)
    )

    tgt = torch.randint(
        0,
        tgt_vocab_size,
        (batch_size, tgt_len)
    )

    tgt_mask = causal_mask(
        tgt_len,
        tgt.device
    )

    logits = model(
        src,
        tgt,
        tgt_mask=tgt_mask
    )

    print("src:", src.shape)
    print("tgt:", tgt.shape)
    print("logits:", logits.shape)