import torch
from torch import nn
import torch.nn.functional as F


def convex_upsample(low_res, weights, scale: int):
    B, C, H, W = low_res.shape

    # 1.Extract 3x3 patches from low_res -> Shape: (B, C, 9, H, W)
    patches = F.unfold(low_res, kernel_size=3, padding=1)
    patches = patches.view(B, C, 9, H, W)

    # 2. Reshaping for broadcasting using the high-res weights
    # We insert dimensions of size 1 for the subpixels (scale)
    patches = patches.view(B, C, 9, H, 1, W, 1)

    # 3. Reshape the weights to spatially separate the subpixel dimensions (scale)
    # weights has the form (B, 9, H*scale, W*scale)
    weights = weights.view(B, 1, 9, H, scale, W, scale)

    # 4. Weighted sum over the 9 neighbors (dim=2)
    out = (patches * weights).sum(dim=2)  # Shape: (B, C, H, scale, W, scale)

    # 5. Convert back to the final, flat high-resolution format
    # The memory layout (H, scale, W, scale) corresponds exactly to (H*scale, W*scale)
    out = out.view(B, C, H * scale, W * scale)

    return out


class SharedConv2d(nn.Module):
    def __init__(self, out_channels, kernel_size=3, padding=0, bias=True):
        super().__init__()
        self.conv2d = nn.Conv2d(
            1, out_channels, kernel_size, padding=padding, bias=bias
        )  # weight sharing over input channels

    def forward(self, x):
        outputs = [self.conv2d(x[:, i : i + 1]) for i in range(x.size(1))]
        return torch.cat(outputs, dim=1)


class ResConn(nn.Module):
    def __init__(self, ch, sqz_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(ch, sqz_ch, 1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(sqz_ch, ch, 1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        res = x
        x = res + self.convs(x)
        return x


class RecResConn(nn.Module):
    def __init__(self, ch, sqz_ch=5):
        super().__init__()
        self.res = ResConn(ch, sqz_ch)

    # Exactly two passes were found to give the best result
    def forward(self, x):
        x = self.res(x)
        x = self.res(x)
        return x


class ConvexUpscaler(nn.Module):
    temperature: float

    def __init__(self, temp: float = 0.5):
        super().__init__()
        self.temperature: float = temp

        self.convs = nn.Sequential(
            SharedConv2d(16, kernel_size=7, padding=3, bias=False),
            nn.GELU(approximate="tanh"),
            nn.GroupNorm(1, 48),
            RecResConn(48, 12),
            nn.GELU(approximate="tanh"),
            nn.GroupNorm(1, 48),
            nn.Conv2d(48, 16, kernel_size=1),
            nn.GELU(approximate="tanh"),
            nn.PixelShuffle(2),
            nn.GroupNorm(1, 4),
            nn.Conv2d(
                4, 9, kernel_size=3, padding=1
            ),  # Ensuring local consistency after the pixel shuffle
        )

    def forward(self, lr):
        # 1. Feature extraction and weight prediction
        x = self.convs(lr)

        # 2. Apply Softmax to the convex combination
        weights = F.softmax(x / self.temperature, dim=1)

        # 3. Upsampling
        interpolated = convex_upsample(lr, weights, 2)

        return interpolated
