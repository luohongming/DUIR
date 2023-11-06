
import torch.nn as nn
import torch.nn.functional as F
import torch
# from archs.common import make_layer
from utils.registry import ARCH_REGISTRY
from archs.convLSTM import ConvLSTMCell
import numpy as np
from archs.Net_Big import Diff_Enc   # 特征生成

class HDA_conv(nn.Module):   # SDL
    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size, attn=False):
        super(HDA_conv, self).__init__()

        self.kernel_size = kernel_size   # 卷积核大小
        # self.attn = attn
        self.num_feat = num_feat   # ？

        # 固定的参数部分  从图像中直接提取出来的特征
        # 输入图像的四维张量[N, C, H, W]
        # Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros')
        # stride（移动的长度） padding（填充与否）groups（是否用分组卷积）
        self.conv_share = nn.Conv2d(num_feat, chan_fixed, kernel_size, stride=1, padding=(kernel_size-1)//2)    # (kernel_size-1)//2 ？
        # 需要修改的参数部分
        self.conv_dyn = conv_dyn(num_feat, chan_dyn, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, out_weight):
        # calculate conv weights
        # out_weight size: [chan_dyn, num_feat, kernel_size, kernel_size]

        # b, c, h, w = x.size()

        out1 = self.conv_dyn(x, out_weight)
        out2 = self.conv_share(x)

        out = torch.cat((out1, out2), dim=1)  # dim维度 在横向叠加  SDL

        return out

# class conv_dyn(nn.Module):
#
#     def forward(self, input, weight):
#         # weight size: [b*out, in ,k, k]
#         b, c, h, w = input.size()
#         out = F.conv2d(input.view(1, -1, h, w), weight, stride=1, padding=(weight.size(2)-1)//2, groups=b*c)
#         return out.view(b, -1, h, w)

class conv_dyn(nn.Conv2d):   # 如何对需要生成的参数  进行卷积

    def _conv_forward(self, input, weight, bias, groups):

        if self.padding_mode != 'zeros':
            # F.pad(input, pad, mode='constant', value=0) 对tensor进行扩充  input:需要扩充的tensor，pad：扩充维度，mode：扩充方法 value：扩充时指定补充值
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            (0, 0), self.dilation, groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, groups)

    def forward(self, input, weight):
        # weight size: [b*out, in ,k, k]
        b, c, h, w = input.size()
        conv_weight = self.weight.repeat(b, 1,1,1)   # repeat()给三个维度各加一行
        out = self._conv_forward(input.view(1, -1, h, w), conv_weight * weight, self.bias, groups=b)    # 最后的输出结果
        return out.view(b, -1, h, w)  # 将一维变为b维

class ResidualBlockNoBN_dyn(nn.Module):  # 系统图中的灰色部分 残差块
    """Residual block without BN.
    It has a style of:
    ::
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, chan_fixed, chan_dyn, num_feat, kernel_size=3, attn=False, res_scale=1.0):
        super(ResidualBlockNoBN_dyn, self).__init__()
        self.res_scale = res_scale
        # 代表灰色的那一块的两个SDLs        先卷积再cat
        self.conv1 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size, attn)
        self.conv2 = HDA_conv(chan_fixed, chan_dyn, num_feat, kernel_size, attn)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, out_weight1, out_weight2):

        identity = x
        out = self.conv1(x, out_weight1)
        # out = self.relu(self.bn1(out))
        out = self.relu(out)
        out = self.conv2(out, out_weight2)
        out = identity + out * self.res_scale
        # out = self.relu(out)
        return out

@ARCH_REGISTRY.register()
class DynEDSR(nn.Module):    # 主架构  定义模型
    def __init__(self,
                 num_in_ch,
                 num_feat=64,
                 num_block=16,
                 res_scale=1,          # ？
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 kernel_size=3,
                 weight_ratio=0.5,
                 attn=False,
                 blind=False,
                 ):

        super(DynEDSR, self).__init__()

        # 下面就一层层开始定义需要用到的函数及方法
        self.img_range = img_range    # 图像范围
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)   # 归一化？
        self.num_feat = num_feat
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)   # 第一个卷积 将输入通道数变为num_feat
        self.kernel_size = kernel_size    # 卷积核大小
        chan_fixed = int(num_feat * weight_ratio)    # 一半固定的参数
        chan_dyn = num_feat - chan_fixed             # 一半需要改变的参数
        self.attn = attn
        self.relu = nn.LeakyReLU(0.1, True)    # 激活函数
        self.blind = blind


        if attn:   # attn非零
            if blind:   # blind非零
                self.embed = Diff_Enc()
                # 加载模型  checkpoint：通用的检查点
                checkpoint = torch.load('./experiments/feature_extract.pth')
                self.embed.load_state_dict(checkpoint)

                # self.embed = Degrade_Encoder()
                # checkpoint = torch.load('/home/lhm/PycharmProject/GIR/experiments/FeatEncoder_v1_96/models/net_g_180000.pth')
                # self.embed.load_state_dict(checkpoint['params'])

                self.conv_represent = nn.Sequential(  # 全连接层 DEM之后的那个全连接层
                    nn.Linear(256, num_feat, bias=False),
                    nn.LeakyReLU(0.1, True),
                    nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False))
            else:

                self.conv_represent = nn.Sequential(   # 计算全连接层的  DAM的输入
                    nn.Linear(3, num_feat, bias=False),
                    nn.LeakyReLU(0.1, True),
                    # nn.Linear(num_feat, chan_dyn*num_feat, bias=False),
                    # nn.LeakyReLU(0.1, True),
                    nn.Linear(num_feat, chan_dyn * num_feat * kernel_size * kernel_size, bias=False)
                )

            # self.forward_lstm = nn.ModuleList()
            # self.backward_lstm = nn.ModuleList()
            # for _ in range(num_block*2):
            #     self.forward_lstm.append(
            #         ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)
            #     )
            #
            #     self.backward_lstm.append(
            #         ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)
            #     )
            # LSTM的过程
            self.forward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)   # 前向
            self.backward_lstm = ConvLSTMCell((kernel_size, kernel_size), num_feat, num_feat, 1, bias=False)  # 后向
            self.lstm_conv_list = nn.ModuleList()
            for _ in range(num_block*2):    # 前后向之后的卷积  前项和后向各来一遍  就得是num_block*2
                self.lstm_conv_list.append(   # 将Conv2d添加进ModuleList列表中
                    nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
                )

        self.num_block = num_block
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(
                ResidualBlockNoBN_dyn(   # 一个灰色（HDA）的输出
                    chan_fixed=chan_fixed,
                    chan_dyn=chan_dyn,
                    num_feat=num_feat,
                    kernel_size=kernel_size,
                    attn=attn,
                    res_scale=res_scale
                ))

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1)       # 变为与最原始的输入图像的channel一样


    def forward(self, x, cond):

        # cond : b, 3
        # b, c, h, w = x.size()

        if self.attn:
            if self.blind:
                # with torch.no_grad():
                if cond.size(1) == 64:
                    represent = cond
                else:
                    # self.embed.eval()
                    # print('realblur')
                    represent, _ = self.embed(x)
                    # _, represent = self.embed(x)

                represent = self.conv_represent(represent)  # b, chan_dyn*num_feat*3*3
                represent = represent.view(-1, self.num_feat, self.kernel_size,
                                           self.kernel_size)  # b*chan_dyn, num_feat, kernel_size, kernel_size
            else:
                represent = self.conv_represent(cond)  # b, chan_dyn*num_feat*3*3
                represent = represent.view(-1, self.num_feat, self.kernel_size,
                                           self.kernel_size)  # b*chan_dyn, num_feat, kernel_size, kernel_size

            h_init_for = torch.zeros_like(represent)
            c_init_for = torch.zeros_like(represent)
            next_state_for = (h_init_for, c_init_for)

            h_init_back = torch.zeros_like(represent)
            c_init_back = torch.zeros_like(represent)
            next_state_back = (h_init_back, c_init_back)

            out_weight_for = []
            out_weight_back = []

            for i in range(self.num_block*2):
                next_state_for = self.forward_lstm(represent, next_state_for)
                out_weight_for.append(next_state_for[0])
                next_state_back = self.backward_lstm(represent, next_state_back)
                out_weight_back.append(next_state_back[0])

            out_weight_back.reverse()

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)

        x1 = x.clone()    # x=x1
        for j in range(self.num_block):
            if self.attn:
                out_weight1 = self.lstm_conv_list[j*2](torch.cat((out_weight_for[j*2], out_weight_back[j*2]), dim=1))
                out_weight2 = self.lstm_conv_list[j*2+1](torch.cat((out_weight_for[j*2+1], out_weight_back[j*2+1]), dim=1))

            else:
                out_weight1 = 1
                out_weight2 = 1

            x1 = self.body[j](x1, out_weight1, out_weight2)

        res = self.conv_after_body(x1)
        res += x
        x = self.conv_last(res)
        x = x / self.img_range + self.mean
        return x
