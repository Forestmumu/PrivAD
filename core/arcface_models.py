

import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample = None, stride = 1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding =0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride

        self.act_fn = nn.ReLU()


    def forward(self, x):
        identity = x.clone()
        x = self.act_fn(self.batch_norm1(self.conv1(x)))

        x = self.act_fn(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)

        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x += identity
        x = self.act_fn(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=False, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, latent_dim, num_channels):

        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.act_fn = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes = 64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes = 128, stride = 2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes = 256, stride = 2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes = 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512* ResBlock.expansion, latent_dim)

        # 模型初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.act_fn(self.batch_norm1(self.conv1(x)))

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

    # 4个层用的子网络模块
    def _make_layer(self, ResBlock, blocks, planes, stride = 1):
        ii_downsample = None
        layers = []

        # 下采样
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers) # 建立Conv1_; Conv2_; Conv3_; Conv4..


def ResNet50(latent_dim, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], latent_dim, channels)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # print('one-hot',one_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1-p) ** self.gamma * logp
        return loss.mean()


if __name__ == '__main__':
    encoder = ResNet50(latent_dim=512, channels=3)
    decoder = ArcMarginProduct(in_features=512,  out_features= 2622)
    label = torch.arange(2)


    x = torch.randn(2, 3, 224, 224)
    z = encoder(x)
    out = decoder(z, label)

    softmax = nn.Softmax(dim=1)
    soft_max_out = softmax(out)

    print('z', z.shape)
    print('soft_max_out', soft_max_out.shape)
