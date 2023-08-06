""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn


class ConvBlock_(nn.Module):
    """BN-ReLU-Conv(k=3x3)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class DenseBlock(nn.Module):
    """Conv * 5 and dense connect"""

    def __init__(self, in_channels, out_channels, k, times=5):
        super(DenseBlock, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(times):
            if i == times - 1:
                self.convs.append(ConvBlock_(in_channels + i * k, out_channels))
            else:
                self.convs.append(ConvBlock_(in_channels + i * k, k))

    def forward(self, x):
        input = x
        outputs = [x]
        for conv in self.convs:
            outputs.insert(0, conv(input))
            input = torch.cat(outputs, 1)
        return outputs[0]


class Down(nn.Module):
    """Encoder arm blocks"""

    def __init__(self, in_channels, out_channels, k, times=5):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            DenseBlock(in_channels, in_channels, k, times),
            ConvBlock_(in_channels, out_channels,
                       kernel_size=1, stride=1, padding=0)
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
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=2)

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
            ConvBlock_(in_channels, out_channels,
                       kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.up = UPB(out_channels, out_channels)

    def forward(self, x1, x2, indices):
        output1 = self.conv(x1)
        output2 = self.up(output1, indices)
        output = torch.cat((output2, x2), 1)
        return output


def get_indices(batch_size, input_channels, input_size, cuda):
    indices = torch.tensor([[[
        [(input_size * j * 2 + i) * 2 for i in range(input_size)]
        for j in range(input_size)
    ]
        for s in range(input_channels)
    ]
        for t in range(batch_size)], dtype=torch.int64)
    if cuda:
        indices = indices.cuda()
    return indices


class Generator(nn.Module):
    """encoder arm plus decoder arm"""

    def __init__(self, init_dims=32,
                 down_scales=[2, 4, 8, 16],
                 image_size=512,
                 dense_k=12, dense_times=5):
        super(Generator, self).__init__()
        self.image_sizes = []
        for i in range(len(down_scales) + 1):
            self.image_sizes.append(image_size // 2 ** (i + 1))
        # down arm(encoder)
        self.extract = nn.Sequential(
            nn.Conv2d(1, init_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(init_dims),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.first_pooling = nn.MaxPool2d(3, stride=2, padding=1)
        self.down_arm = nn.ModuleList()
        self.down_pooling = nn.MaxPool2d(2, stride=2)
        for i, scale in enumerate(down_scales):
            if i == 0:
                self.down_arm.append(Down(init_dims,
                                              scale * init_dims, dense_k))
            else:
                self.down_arm.append(Down(down_scales[i - 1] * init_dims,
                                              scale * init_dims, dense_k))

        # up arm(decoder)
        self.up_arm = nn.ModuleList()
        for i, scales in enumerate(down_scales[::-1]):
            if i == 0:
                self.up_arm.append(
                    Up(scales * init_dims, scales * init_dims))
            else:
                self.up_arm.append(
                    Up(2 * down_scales[-i] * init_dims, scales * init_dims))
        self.up_arm.append(Up(2 * down_scales[0] * init_dims, init_dims))
        self.reconstruct = nn.Sequential(
            nn.BatchNorm2d(2 * init_dims),
            nn.Conv2d(2 * init_dims, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.on_cuda = False

    def forward(self, x, show_internal=False):
        # encoder arm
        internal_image = None
        if show_internal:
            internal_image = []
            internal_image.append(x[0, 0, ...])
        feature = self.extract(x)
        outputs = [feature]
        if show_internal:
            internal_image.append(feature[0, 0, ...])
        pooled = self.first_pooling(feature)
        for down in self.down_arm:
            output = down(pooled)
            outputs.insert(0, output)
            if show_internal:
                internal_image.append(output[0, 0, ...])
            pooled = self.down_pooling(output)
        up_input = pooled

        # decoder arm
        if feature.is_cuda:
            self.on_cuda = True
        for i, (output, up, img_size) in enumerate(zip(outputs, self.up_arm, self.image_sizes[::-1])):
            if i == 0:
                indices = get_indices(
                    output.shape[0], output.shape[1], img_size, self.on_cuda)
            else:
                indices = get_indices(
                    up_input.shape[0], up_input.shape[1] // 4, img_size, self.on_cuda)
            up_input = up(up_input, output, indices)
            if show_internal:
                internal_image.append(up_input[0, 0, ...])
        # print(up_result5.shape)
        result = self.reconstruct(up_input)
        if show_internal:
            internal_image.append(result[0, 0, ...])
        # print(result.shape)
        if show_internal:
            return result, internal_image
        else:
            return result
