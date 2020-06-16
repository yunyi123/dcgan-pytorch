'''this code was desinged by nike hu'''

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from face_model import Discrimite, Generate # 这里face_model是搭建模型的py文件的名字
import visdom
from torchvision.datasets import ImageFolder
from torch import nn, autograd
from torch.autograd import Variable

batchsize = 64

def getData():
    trainData = ImageFolder('F:/code/DataSet/data/focusight1_round1_train_part1/OK_Images', transform=transforms.Compose([
     transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])) # 使用transform.Compose(),里面要加上[]，否则会报错，而且无法迭代数据，这里是加载训练图片路径的
    trainLoader = DataLoader(trainData, batch_size=batchsize, shuffle=True, drop_last=True)

    return trainLoader


# 这里的函数是根据wgan的原理设计的，目的是让鉴别器的梯度变化在1附近
def gradient_penalty(D, xr, xf):
    batch = xr.size(0)
    t = torch.rand(batch, 1, 1, 1).cuda()
    t = t.expand_as(xr)
    mid = t * xr + (1 - t) * xf # 线性差值
    mid.requires_grad_() # 设置上面的线性差值需要求导
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred), create_graph=True,
                          retain_graph=True, only_inputs=True)[0]
    # 如果输入x，输出是y，则求y关于x的导数（梯度）：
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean() # 二范数和1的差值的平方的均值
    return gp

def trainModel():
    torch.manual_seed(23) # 随机种子设置
    generate_net = Generate()
    print(generate_net)
    discrimi_net = Discrimite()
    print(discrimi_net)
    device = torch.device('cuda:0')
    generate_net = generate_net.to(device)
    discrimi_net = discrimi_net.to(device)
    generate_optimer = torch.optim.Adam(generate_net.parameters(), lr=0.0002, betas=(0.5, 0.9))
    discrimi_optimer = torch.optim.Adam(discrimi_net.parameters(), lr=0.0002, betas=(0.5, 0.9))
    trainLoader = getData()
    viz = visdom.Visdom()
    critimer = nn.BCELoss()
    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    epoch = 0
    for i in range(10000):
        print('------------------------------第', i, '次的函数统计----------------------------------------------------')
        for really_x, _ in trainLoader:
            for _ in range(1): # 这里可以设置为先训练鉴别器多次然后再训练生成器，只需要把1改一下，然后把下面的一行代码恢复
                # really_x = next(iter(trainLoader))[0]
                really_x = really_x.to(device)
                batchs = really_x.size(0)
                pred_r = discrimi_net(really_x).view(batchs, -1) # 生成器生成的数据,最后维度是[batch, 1]


                real_label = Variable(torch.ones((batchs, 1)), requires_grad=False)
                fake_label = Variable(torch.zeros((batchs, 1)), requires_grad=False)
                real_label = real_label.to(device)
                fake_label = fake_label.to(device)
                loss_r = critimer(pred_r, real_label)
                fake_x = torch.randn((batchs, 100, 1, 1))
                fake_x = fake_x.to(device) # 放到gpu上
                fake_x = generate_net(fake_x).detach()
                pred_f = discrimi_net(fake_x).view(batchs, -1) # 生成的照片还要进行判别

                loss_f = critimer(pred_f, fake_label)

                gp = gradient_penalty(discrimi_net, really_x, fake_x.detach())
                loff_D = loss_r + loss_f + 0.2*gp # 这里的0.2是一个可以变化的参数，数值不一样，可能最后效果不一样
                discrimi_optimer.zero_grad()
                loff_D.backward()
                discrimi_optimer.step()
            # 接下来是生成器的loss
            fake_x1 = torch.randn((batchs, 100, 1, 1))
            fake_x1 = fake_x1.to(device)
            fake_image2 = generate_net(fake_x1)
            fake_data2 = discrimi_net(fake_image2).view(batchs, -1)

            real_label = Variable(torch.ones((batchs, 1)), requires_grad=False)
            real_label = real_label.to(device)
            generate_losses = critimer(fake_data2, real_label)

            generate_optimer.zero_grad()
            generate_losses.backward()
            generate_optimer.step()
            print('第', i, '个', '生成器的loss->', generate_losses.item(), '判断器的loss->', loff_D.item())
            viz.images(fake_image2, nrow=8, win='x', opts=dict(title='x'))
            viz.line([[loff_D.item(), generate_losses.item()]], [epoch], win='loss', update='append')
            epoch += 1
        # torch.save(generate_net, '/content/drive/My Drive/model/generate.pkl')
        # torch.save(discrimi_net, '/content/drive/My Drive/model/discrimi.pkl')


if __name__ == '__main__':
    trainModel()
