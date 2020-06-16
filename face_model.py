'''this code was desinged by nike hu'''
import torch
import torch.nn as nn

device = torch.device('cuda:0')



# 鉴定器网络
class Discrimite(nn.Module):
    def __init__(self):
        super(Discrimite, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False), # 这里false就是bias不进行梯度更新
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.weight_init()

    def forward(self, x):
        x = self.conv(x)
        return x

    # 初始化参数
    def weight_init(self):
        for m in self.conv.modules():
            if isinstance(m, nn.ConvTranspose2d): # 判断一个变量是否是某个类型可以用isinstance()判断
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02) # 初始化为正太分布，torch.nn.init.uniform_(tensor, a=0, b=1)是均匀分布
                nn.init.constant_(m.bias.data, 0) # 初始化为常数


class Generate(nn.Module):
    def __init__(self):
        super(Generate, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.weight_init()

    def forward(self, x):
        x = self.conv(x)
        return x

    # 初始化参数
    def weight_init(self):
        for m in self.conv.modules():
            if isinstance(m, nn.ConvTranspose2d): # 判断一个变量是否是某个类型可以用isinstance()判断
                nn.init.normal_(m.weight.data, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0) # 将bias的值设置为0





if __name__ == '__main__':
    # x = torch.rand(64, 100, 1, 1)
    # x = x.cuda()
    # net = Generate()
    # net.to(device)
    # x = net(x)
    # print(x.shape)
    x = torch.randn(16, 3, 64, 64)
    x = x.to(device)
    net = Discrimite()
    net.to(device)
    x = net(x)
    print(x.shape)