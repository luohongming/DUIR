import torch.nn as nn
import torch
import math
import torchvision.models as models


class BackboneRes(nn.Module):
    def __init__(self, layer=50):
        super(BackboneRes, self).__init__()

        if layer == 50:
            self.backbone = models.resnet50(pretrained=True)
        elif layer == 34:
            self.backbone = models.resnet34(pretrained=True)
        elif layer == 101:
            self.backbone = models.resnet101(pretrained=True)
        elif layer == 152:
            self.backbone = models.resnet152(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)


    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        out = self.backbone.layer4(x)

        return out


class BackboneDense(nn.Module):
    def __init__(self, layer=121):
        super(BackboneDense, self).__init__()

        if layer == 121:
            self.backbone = models.densenet121(pretrained=True).features
        elif layer == 201:
            self.backbone = models.densenet201(pretrained=True).features
        elif layer == 161:
            self.backbone = models.densenet161(pretrained=True).features
        else:
            self.backbone = models.densenet169(pretrained=True).features

        # self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):

        x = self.backbone.conv0(x)
        x = self.backbone.norm0(x)
        x = self.backbone.relu0(x)
        x = self.backbone.pool0(x)
        x1 = self.backbone.transition1(self.backbone.denseblock1(x))
        x2 = self.backbone.transition2(self.backbone.denseblock2(x1))
        x3 = self.backbone.transition3(self.backbone.denseblock3(x2))
        x4 = self.backbone.norm5(self.backbone.denseblock4(x3))
        # out = self.gap(x4)
        out = x4

        # return out.view(out.size(0), -1)
        return out


class Residual_Block(nn.Module):
    def __init__(self, in_channels=32):
        super(Residual_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        identity_data = x
        out = self.conv(x)
        final = out + identity_data

        return final


class Residual_Block_Enc(nn.Module):
    def __init__(self, in_channels=32, double=True):
        super(Residual_Block_Enc, self).__init__()

        if double:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                          bias=True),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
            )

    def forward(self, x):
        identity_data = x
        out = self.conv(x)
        final = out + identity_data

        return final



class Cont_Enc(nn.Module):
    def __init__(self):
        super(Cont_Enc, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.residual = self.make_layer(Residual_Block, 4, 256)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layer(self, block, num, in_channels=32):
        layers = []
        for i in range(num):
            layers.append(block(in_channels=in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        o1 = self.conv1(x)
        o2 = self.conv2(o1)

        out = self.residual(o2)

        return out


class SiPNet(nn.Module):
    def __init__(self, dim_in):
        super(SiPNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim_in * 21, 64),
            # nn.LeakyReLU(),
            nn.ReLU(),
        )

    def forward(self, x):

        o1 = self.conv1(x)
        num, c, h, w = o1.size()

        for i in range(4):  # SPP
            level = i + 1
            if level != 3:
                kernel_size = (math.ceil(h / level), math.ceil(w / level))
                stride = (math.ceil(h / level), math.ceil(w / level))
                pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
                # print(pooling)

                pooling_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pooling)
                res = pooling_layer(o1)

                if i == 0:
                    x_flatten = res.view(num, -1)
                else:
                    x_flatten = torch.cat([x_flatten, res.view(num, -1)], dim=1)

        final = self.fc1(x_flatten)

        return final


class Cont_G(nn.Module):
    def __init__(self):
        super(Cont_G, self).__init__()

        self.residual = self.make_layer(Residual_Block, 4, 256)


        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(16),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layer(self, block, num, in_channels=32):
        layers = []
        for i in range(num):
            layers.append(block(in_channels=in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        o1 = self.residual(x)

        o2 = self.dec1(o1)
        out = self.dec2(o2)

        return out


class RegNet(nn.Module):
    def __init__(self, clen=4):
        super(RegNet, self).__init__()

        if clen != 0:
            self.pool = nn.Sequential(
                nn.Conv2d(1024, clen, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1)),
            )

        self.fc = nn.Sequential(
            nn.Linear(256+clen, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )
        self.clen = clen

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, xc):
        
        if self.clen != 0:
            xcf = self.pool(xc)
            feat = torch.cat([x, xcf.reshape([xcf.size(0), -1])], dim=1)
        else:
            feat = x

        out = self.fc(feat)

        return out

class RegNetAu(nn.Module):
    def __init__(self, alen=16):
        super(RegNetAu, self).__init__()

        # self.pool = nn.Sequential(
        #     nn.Conv2d(1024, alen, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1)),
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, alen),
        )

        self.fc = nn.Sequential(
            nn.Linear(258+alen, 16),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(16, 16),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, xc):

        # xcf = self.pool(xc)
        # feat = torch.cat([x, xcf.reshape([xcf.size(0), -1])], dim=1)

        xcf = self.fc1(xc)
        feat = torch.cat([x, xcf], dim=1)

        out = self.fc(feat)

        return out

class RegNetv2(nn.Module):
    def __init__(self, dlen=256):
        super(RegNetv2, self).__init__()

        self.dlen = dlen
        if dlen > 128:
            inode = 256
            onode = dlen
        else:
            inode = dlen
            onode = 128

        self.fc = nn.Sequential(
            nn.Linear(inode, onode),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(onode, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        # print(int(256/self.dlen))
        if self.dlen > 128:
            feat = x
        else:
            feat = x[:,0:-1:int(256/self.dlen)]

        out = self.fc(feat)

        return out


class MisINSResBlock(nn.Module):

    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.BatchNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.BatchNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
          model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z, mask):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        z_mask = z_expand * mask
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_mask], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_mask], dim=1))
        out += residual
        return out


def gaussian_weights_init(m):

      classname = m.__class__.__name__

      if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class Dist_G(nn.Module):
    def __init__(self):
        super(Dist_G, self).__init__()

        self.pool1 = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.ReLU(False),
            # nn.Linear(256, 1024),
            nn.ReLU(False),
        )

        self.decA1 = MisINSResBlock(256, 256)
        self.decA2 = MisINSResBlock(256, 256)
        self.decA3 = MisINSResBlock(256, 256)
        self.decA4 = MisINSResBlock(256, 256)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(16),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layer(self, block, num, in_channels=32):
        layers = []
        for i in range(num):
            layers.append(block(in_channels=in_channels))
        return nn.Sequential(*layers)

    def forward(self, x, z, mask):

        oz = self.pool1(z)
        z1, z2, z3, z4 = torch.split(oz, 256, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()

        out1 = self.decA1(x, z1, mask)
        out2 = self.decA2(out1, z2, mask)
        out3 = self.decA3(out2, z3, mask)
        out4 = self.decA4(out3, z4, mask)

        out5 = self.dec1(out4)
        final = self.dec2(out5)

        return final



class Diff_Enc(nn.Module):  # This is a diff_encoder with several jump
    def __init__(self):
        super(Diff_Enc, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.residual2 = self.make_layer(Residual_Block_Enc, 1, 64, double=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.residual3 = self.make_layer(Residual_Block_Enc, 1, 64, double=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.residual4 = self.make_layer(Residual_Block_Enc, 1, 64, double=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.residual5 = self.make_layer(Residual_Block_Enc, 1, 64, double=False)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.mask1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.sp1 = SiPNet(64)
        self.sp2 = SiPNet(64)
        self.sp3 = SiPNet(64)
        self.sp4 = SiPNet(64)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layer(self, block, num, in_channels=32, double=True):
        layers = []
        for i in range(num):
            layers.append(block(in_channels=in_channels, double=double))
        return nn.Sequential(*layers)

    def forward(self, x):

        o1 = self.conv1(x)
        o2 = self.conv2(self.residual2(o1))
        f1 = self.sp1(o2)
        o3 = self.conv3(self.residual3(nn.functional.max_pool2d(o2, [2, 2])))
        f2 = self.sp2(o3)
        o4 = self.conv4(self.residual4(nn.functional.max_pool2d(o3, [2, 2])))
        f3 = self.sp3(o4)
        o5 = self.conv5(self.residual5(nn.functional.max_pool2d(o4, [2, 2])))
        f4 = self.sp4(o5)

        feat = torch.cat([f1, f2, f3, f4], dim=1)

        mask = self.mask1(x)
        # mask = torch.FloatTensor(mask.shape).fill_(1.0).cuda()  # We don't use mask at the first time!

        return feat, mask
