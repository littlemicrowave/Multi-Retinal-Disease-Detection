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
    def __init__(self, H, W, channels, num_heads, projection_dim = None, use_rpe = True, max_rpe_dist = 4, dropout = 0.2, mask=None):
        super().__init__()
        if projection_dim == None:
            projection_dim = channels
        if projection_dim % num_heads != 0:
            raise "Projection number of dimensions is indisible by number of heads!"
        self.use_rpe = use_rpe
        self.projection_dim = projection_dim
        self.input_dim = channels
        self.num_heads = num_heads
        self.head_depth = projection_dim // num_heads
        self.mask = mask
        self.attn_denum = self.head_depth ** (-0.5)
        self.xproj = nn.Linear(channels, projection_dim)
        self.wq = nn.Linear(projection_dim, projection_dim)
        self.wk = nn.Linear(projection_dim, projection_dim)
        self.wv = nn.Linear(projection_dim, projection_dim)
        self.wo = nn.Linear(projection_dim, channels)
        self.scale_factor = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.norm = nn.LayerNorm(self.input_dim)
        if use_rpe:
            self.rpe = RelativePositionalEncoding(H, W, max_rpe_dist, self.head_depth)
        else: 
            self.pe = nn.Parameter(torch.randn(1, W*H, projection_dim), requires_grad = True)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v):
        # q,k,v: (B, H, L, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_denum
        if self.use_rpe:
            Rk = self.rpe.get_k()                      # (L, L, D)
            pos_scores = torch.einsum("bhld,lmd->bhlm", q, Rk)
            scores = scores + pos_scores

        if self.mask is not None:
            scores = scores.masked_fill(self.mask, -torch.inf)

        attn = self.softmax(scores)

        out = torch.matmul(attn, v)

        if self.use_rpe:
            Rv = self.rpe.get_v()                      # (L, L, D)
            pos_out = torch.einsum("bhlm,lmd->bhld", attn, Rv)
            out = out + pos_out

        return out

    def forward(self, x):
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        seq_len = h * w

        #flatten spatial dimensions
        x = torch.permute(x, [0, 2, 3, 1])
        x = torch.reshape(x, [b, seq_len, c])
        x_ = self.xproj(x)
        if not self.use_rpe:
            x_ = x_ + self.pe

        q = self.wq(x_)
        k = self.wk(x_)
        v = self.wv(x_)

        #split heads
        q = torch.reshape(q, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        k = torch.reshape(k, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        v = torch.reshape(v, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)

        #attention compute
        out = self.attention(q, k, v)
        #merge heads
        out = out.transpose(1, 2)
        out = out.reshape(b, seq_len, self.projection_dim)
        out = self.wo(out)
        out = self.dropout(out)

        #add residual and scale
        out = x + out * self.scale_factor
        out = self.norm(out)
        #permute and reshape
        out = torch.permute(out, [0, 2, 1])
        out = torch.reshape(out, [b, c, h, w])
        return out

class RelativePositionalEncoding(nn.Module):
    def __init__(self, h, w, max_distance, head_depth):
        super().__init__()

        # Shaw et al.
        self.rel_x_k = nn.Embedding(2*max_distance + 1, head_depth)
        self.rel_y_k = nn.Embedding(2*max_distance + 1, head_depth)
        self.rel_x_v = nn.Embedding(2*max_distance + 1, head_depth)
        self.rel_y_v = nn.Embedding(2*max_distance + 1, head_depth)

        # offsets 
        ys = torch.arange(h * w) // w
        xs = torch.arange(h * w) %  w
        dx = xs.unsqueeze(1) - xs
        dy = ys.unsqueeze(1) - ys
        dx = torch.clip(dx, -max_distance, max_distance) + max_distance
        dy = torch.clip(dy, -max_distance, max_distance) + max_distance
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)


    def get_k(self):
        return self.rel_x_k(self.dx) + self.rel_y_k(self.dy)  # (L, L, D)

    def get_v(self):
        return self.rel_x_v(self.dx) + self.rel_y_v(self.dy)  # (L, L, D)
    