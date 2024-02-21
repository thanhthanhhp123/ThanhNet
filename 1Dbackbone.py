import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=dilation_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=dilation_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # Thêm residual vào đầu ra
        out = self.relu(out)
        return out

class ResNet1DCNN(nn.Module):
    def __init__(self):
        super(ResNet1DCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1) 
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, kernel_size=3, dilation_rate=1),
            ResidualBlock(64, 64, kernel_size=3, dilation_rate=2),
            ResidualBlock(64, 64, kernel_size=3, dilation_rate=4)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x)
        return x


if __name__ == "__main__":
    model = ResNet1DCNN()
    summary(model, (80, 57))