import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        hidden = max(1, ch//ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c) # squeeze
        y = self.fc(y).view(b, c, 1, 1) # excitation

        return x * y # scale the channels


class MultiHeadAttentionCNN(nn.Module):
    def __init__(self, input_dim, num_heads, mask=None):
        super().__init__()
        if input_dim % num_heads != 0:
            raise "Projection number of dimensions is indisible by number of heads!"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_depth = input_dim // num_heads
        self.mask = mask
        self.attn_denum = self.head_depth ** (-0.5)

        self.wq = nn.Linear(input_dim, input_dim)
        self.wk = nn.Linear(input_dim, input_dim)
        self.wv = nn.Linear(input_dim, input_dim)
        #self.wo = nn.Linear(input_dim, input_dim) #not used
        self.scale_factor = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.norm = nn.LayerNorm(self.input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v):
        # q, k, v: (b, heads, seq_len, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.attn_denum

        if self.mask is not None:
            scores = torch.masked_fill(scores, self.mask, -torch.inf)

        scores = self.softmax(scores)
        return torch.matmul(scores, v)

    def forward(self, x):
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        seq_len = h * w

        #flatten spatial dimensions
        x = torch.permute(x, [0, 2, 3, 1])
        x = torch.reshape(x, [b, seq_len, c])

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        #split heads
        q = torch.reshape(q, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        k = torch.reshape(k, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        v = torch.reshape(v, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)

        #attention compute
        out = self.attention(q, k, v)
        #merge heads
        out = out.transpose(1, 2)
        out = out.reshape(b, seq_len, c)
        #out = self.wo(out)
        #add residual and scale
        out = out * self.scale_factor + x
        out = self.norm(out)
        #permute and reshape
        out = torch.permute(out, [0, 2, 1])
        out = torch.reshape(out, [b, c, h, w])
        return out

