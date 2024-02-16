import torch
import torch.nn as nn
import math
class res_block(nn.Module):
    def __init__(self, channels = 64):
        super(res_block, self).__init__()
        self.basic_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )    
    
    def forward(self, x):
        return self.basic_block(x) + x

class dis_block(nn.Module):
    def __init__(self, input, output, stride):
        super().__init__()
        self.basic_block = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.basic_block(x)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residual =  nn.Sequential(
            res_block(),
            res_block(),
            res_block(),
            res_block(),
            res_block()
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1,padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv4 =  nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1,padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv5 = nn.Conv2d(64, 3, kernel_size=9, stride=1,padding=4)
        
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.residual(out1)
        out2 = self.conv2(out2)
        out3 = self.conv3(out2 + out1)
        out3 = self.conv4(out3)
        out3 = self.conv5(out3)
        return out3
    
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.down =  nn.Sequential(
            dis_block(64, 64, 2),
            dis_block(64, 128, 1),
            dis_block(128, 128, 2),
            dis_block(128, 256, 1),
            dis_block(256, 256, 2),
            dis_block(256, 512, 1),
            dis_block(512, 512, 2)
        )
        self.den =  nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.down(out1)
        out1 = self.den(out1)
        return out1
    

