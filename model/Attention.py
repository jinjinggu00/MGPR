import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)


    def forward(self, x):

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x_sa = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x_sa)
        output = self.softmax(x1)
        att = x * output

        return att

class channel_attention(nn.Module):
    def __init__(self, channels, retio = 16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels//retio, False),
            nn.ReLU(),
            nn.Linear(channels//retio, channels, False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        print(out)

        return out * x
