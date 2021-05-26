"""
16-layer U-Net model
"""
import torch.nn as nn

from .mobilenetv3 import get_mobilenet_parts
from models.unet_tools import ConvSig
from models.unet_tools import FCNN
from models.unet_tools import UNetDown
from models.unet_tools import UNetUp


class UNet(nn.Module):
    def __init__(self, kernel_size=3, skip=True):
        super().__init__()
        self.skip = skip

    def forward(self, inp):
        """
        Args:
            inp (tensor) : Tensor of input minibatch

        Returns:
            (tensor): Change detection output
            (tensor): Domain output. Will not be returned when self.adversarial="no"
        """
        # print("inp", inp.shape) # [8, 9, 224, 224]
        d1 = self.enc1(inp)
        # print("d1", d1.shape) # [8, 16, 112, 112]
        d2 = self.enc2(d1)
        # print("d2", d2.shape) # [8, 16, 56, 56]
        d3 = self.enc3(d2)
        # print("d3", d3.shape) # [8, 24, 28, 28]
        d4 = self.enc4(d3)
        # print("d4", d4.shape) # [8, 40, 14, 14]
        d5 = self.enc5(d4)
        # print("d5", d5.shape) # [8, 96, 7, 7]

        # Mobilenet
        # inp torch.Size([8, 9, 224, 224])
        # d1 torch.Size([8, 9, 224, 224])
        # d2 torch.Size([8, 16, 112, 112])
        # d3 torch.Size([8, 16, 56, 56])
        # d4 torch.Size([8, 24, 28, 28])
        # d5 torch.Size([8, 48, 14, 14])


        # VGG 16
        # inp torch.Size([8, 9, 224, 224])
        # d1 torch.Size([8, 64, 224, 224])
        # d2 torch.Size([8, 128, 112, 112])
        # d3 torch.Size([8, 256, 56, 56])
        # d4 torch.Size([8, 512, 28, 28])
        # d5 torch.Size([8, 512, 14, 14])


        if self.skip:
            u4 = self.dec4(d5, d4)
            # print("u4", u4.shape)
            u3 = self.dec3(u4, d3)
            # print("u3", u3.shape)
            u2 = self.dec2(u3, d2)
            # print("u2", u2.shape)
            u1 = self.dec1(u2, d1)
            # print("u1", u1.shape)

        else:
            u4 = self.dec4(d5)
            u3 = self.dec3(u4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)

        cd_out = self.out(u1)
        return cd_out


class UNetMobilenetv3(UNet):
    def __init__(self, input_channel: int, kernel_size=3, skip=True):
        super().__init__(kernel_size, skip)

        cfgs = [
            # k, t, c, SE, HS, s
            [3,    1,  16, 1, 0, 2],
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1],
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            # [5,    6,  96, 1, 1, 2],
            # [5,    6,  96, 1, 1, 1],
            # [5,    6,  96, 1, 1, 1],
        ]
        layers = get_mobilenet_parts(cfgs, input_channel, "small", width_mult=1.)
        # layers should be a list of length 1 + len(cfgs) = 12
        self.enc1 = nn.Sequential(nn.Identity())
        self.enc2 = nn.Sequential(layers[0])
        self.enc3 = nn.Sequential(layers[1])
        self.enc4 = nn.Sequential(*layers[2:4])
        self.enc5 = nn.Sequential(*layers[4:9])
        # self.enc5 = nn.Sequential(*layers[5:10])

        self.dec4 = UNetUp(48, skip*24, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip*16, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip*16, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip*9, 64, 2, batch_norm=True, kernel_size=kernel_size)

        self.out = ConvSig(64)

        print("enc1")
        print(self.enc1)
        print("enc2")
        print(self.enc2)
        print("enc3")
        print(self.enc3)
        print("enc4")
        print(self.enc4)
        print("enc5")
        print(self.enc5)

class unet_mobilenetv3(UNetMobilenetv3):
    pass

class UNetMobilenetv3Small(UNet):
    def __init__(self, input_channel: int, kernel_size=3, skip=True):
        super().__init__(kernel_size, skip)

        cfgs = [
            # k, t, c, SE, HS, s
            [3,    1,  16, 1, 0, 2],
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1],
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            # [5,    6,  96, 1, 1, 2],
            # [5,    6,  96, 1, 1, 1],
            # [5,    6,  96, 1, 1, 1],
        ]
        layers = get_mobilenet_parts(cfgs, input_channel, "small", width_mult=1.)
        # layers should be a list of length 1 + len(cfgs) = 12
        self.enc1 = nn.Sequential(nn.Identity())
        self.enc2 = nn.Sequential(layers[0])
        self.enc3 = nn.Sequential(layers[1])
        self.enc4 = nn.Sequential(*layers[2:4])
        self.enc5 = nn.Sequential(*layers[4:9])
        # self.enc5 = nn.Sequential(*layers[5:10])

        self.dec4 = UNetUp(48, skip*24, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(128, skip*16, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(64, skip*16, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(64, skip*9, 32, 2, batch_norm=True, kernel_size=kernel_size)

        self.out = ConvSig(32)

        print("enc1")
        print(self.enc1)
        print("enc2")
        print(self.enc2)
        print("enc3")
        print(self.enc3)
        print("enc4")
        print(self.enc4)
        print("enc5")
        print(self.enc5)

class UNetVgg16(UNet):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """

    def __init__(self, inp_ch, kernel_size=3, skip=True):
        super().__init__(kernel_size, skip)
        self.enc1 = UNetDown(inp_ch, 64, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(64, 128, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(256, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(512, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)

        self.dec4 = UNetUp(512, skip*512, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip*256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip*128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)

        self.out = ConvSig(64)

class unet_vgg16(UNetVgg16):
    pass

class unet16(UNetVgg16):
    pass

class UNetVgg16Small(UNet):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """

    def __init__(self, inp_ch, kernel_size=3, skip=True):
        super().__init__(kernel_size, skip)
        self.enc1 = UNetDown(inp_ch, 32, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(32, 64, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(64, 64, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(64, 128, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(128, 128, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)

        self.dec4 = UNetUp(128, skip*128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(128, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(64, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(64, skip*32, 32, 2, batch_norm=True, kernel_size=kernel_size)

        self.out = ConvSig(32)
