import torch
import torch.nn.functional as F
from math import exp

device = 'cuda'

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

#win = create_window(window_size=11, channel=12)
#print(win.shape)


def ssim_loss(x, y, window_size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    计算两个图像之间的结构相似性损失函数。

    Args:
        x: 第一个图像，形状为(N,C,H,W)。
        y: 第二个图像，形状为(N,C,H,W)。
        window_size: 窗口大小，用于计算局部结构相似性。默认为11。
        sigma: 高斯核标准差，用于平滑图像。默认为1.5。
        c1: 常数，用于避免分母为0的情况。默认为(0.01**2)。
        c2: 常数，用于避免分母为0的情况。默认为(0.03**2)。

    Returns:
        ssim_loss: 计算得到的结构相似性损失函数值。
    """
    # 平滑图像
    #window = torch.Tensor(torch.ones(12, 1, window_size, window_size)).to(x.device)
    window = create_window(window_size=11, channel=3).to(device)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.shape[1])
    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=x.shape[1]) - mu_x ** 2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=y.shape[1]) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.shape[1]) - mu_x * mu_y

    # 计算结构相似性
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_loss = torch.mean(numerator / denominator)

    return 1 - ssim_loss
