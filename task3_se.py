
from torch import nn

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