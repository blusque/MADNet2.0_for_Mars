import torch
import torch.nn as nn


class ConvBlock_(nn.Module):
    """BN-ReLU-Conv(k=3x3)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels
                      , kernel_size=kernel_size
                      , stride=stride
                      , padding=padding)
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class DenseBlock(nn.Module):
    """Conv * 5 and dense connect"""

    def __init__(self, in_channels, out_channels, k):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvBlock_(in_channels, k)
        self.conv2 = ConvBlock_(in_channels + k * 1, k)
        self.conv3 = ConvBlock_(in_channels + k * 2, k)
        self.conv4 = ConvBlock_(in_channels + k * 3, k)
        self.conv5 = ConvBlock_(in_channels + k * 4, out_channels)

    def forward(self, x):
        input1 = x
        output1 = self.conv1(input1)
        input2 = torch.cat((output1, x), 1)
        output2 = self.conv2(input2)
        input3 = torch.cat((output2, output1, x), 1)
        output3 = self.conv3(input3)
        input4 = torch.cat((output3, output2, output1, x), 1)
        output4 = self.conv4(input4)
        input5 = torch.cat((output4, output3, output2, output1, x), 1)
        output5 = self.conv5(input5)
        return output5


class Down(nn.Module):
    """Encoder arm blocks"""

    def __init__(self, in_channels, out_channels, k):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            DenseBlock(in_channels, in_channels, k),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        output = self.down(x)
        return output


class UPB(nn.Module):
    """Unpooling-Conv-ReLU-Conv-ReLU
                |_____Conv_____|
    """

    def __init__(self, in_channels, out_channels):
        super(UPB, self).__init__()
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(1e-2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x, indices):
        unpooled = self.up(x, indices)
        i = self.shortcut(unpooled)
        f = self.conv(unpooled)
        output = i + f
        return output


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2)
        )
        self.up = UPB(out_channels, out_channels)

    def forward(self, x1, x2, indices):
        output1 = self.conv(x1)
        output2 = self.up(output1, indices)
        output = torch.cat((output2, x2), 1)
        return output
