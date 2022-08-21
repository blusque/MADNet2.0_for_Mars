""" Full assembly of the parts to form the complete network """
from .generator_parts import *


def get_indices(batch_size, input_channels, input_size):
    indices = torch.tensor([[
        [
            [(input_size * j * 2 + i) * 2 for i in range(input_size)]
            for j in range(input_size)
        ]
        for s in range(input_channels)
    ]
        for t in range(batch_size)])
    return indices


class Generator(nn.Module):
    """encoder arm plus decoder arm"""

    def __init__(self):
        super(Generator, self).__init__()
        # down arm(encoder)
        self.extract = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(1e-2)
        )
        self.first_pooling = nn.MaxPool2d(3, stride=2, padding=1)
        self.down1 = Down(64, 64, 12)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.down2 = Down(64, 128, 12)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.down3 = Down(128, 256, 12)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.down4 = Down(256, 512, 12)
        self.pooling4 = nn.MaxPool2d(2, stride=2)

        # up arm(decoder)
        self.up1 = Up(512, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.up5 = Up(128, 64)
        self.reconstruct = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2)
        )

    def forward(self, x):
        # encoder arm
        feature = self.extract(x)
        first_pooled = self.first_pooling(feature)
        output1 = self.down1(first_pooled)
        pooled1 = self.pooling1(output1)
        output2 = self.down2(pooled1)
        pooled2 = self.pooling2(output2)
        output3 = self.down3(pooled2)
        pooled3 = self.pooling3(output3)
        output4 = self.down4(pooled3)  # 16x512x16x16
        encoder_result = self.pooling4(output4)

        # decoder arm
        indices1 = get_indices(output4.shape[0]
                               , output4.shape[1], 8)
        up_result1 = self.up1(encoder_result, output4, indices1)
        print(up_result1.shape)
        indices2 = get_indices(up_result1.shape[0]
                               , up_result1.shape[1] // 4, 16)
        up_result2 = self.up2(up_result1, output3, indices2)
        print(up_result2.shape)
        indices3 = get_indices(up_result2.shape[0]
                               , up_result2.shape[1] // 4, 32)
        up_result3 = self.up3(up_result2, output2, indices3)
        print(up_result3.shape)
        indices4 = get_indices(up_result3.shape[0]
                               , up_result3.shape[1] // 4, 64)
        up_result4 = self.up4(up_result3, output1, indices4)
        print(up_result4.shape)
        indices5 = get_indices(up_result4.shape[0]
                               , up_result4.shape[1] // 2, 128)
        up_result5 = self.up5(up_result4, feature, indices5)
        print(up_result5.shape)
        result = self.reconstruct(up_result5)
        print(result.shape)
        return result
