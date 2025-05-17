import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.Unet_Fea_Fusion import Unet
from models.modules.Subnet_constructor import DenseBlock
import torch.nn.init as init
from einops import rearrange
import numbers
import models.modules.module_util as mutil
import torch.nn.functional as F
from models.modules.mwcnn.mwcnn import MWCNN


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarUping(nn.Module):
    def __init__(self, channel_in):
        super(HaarUping, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class DB4Uping(nn.Module):
    def __init__(self, channel_in):
        super(DB4Uping, self).__init__()
        self.channel_in = channel_in

        # DB4 滤波器系数 (归一化)
        db4_low = torch.tensor([0.48296, 0.83652, 0.22414, -0.12941], dtype=torch.float32)  # 低通
        db4_high = torch.tensor([-0.12941, -0.22414, 0.83652, -0.48296], dtype=torch.float32)  # 高通

        # 构建4个方向的2D滤波器权重 [4, 1, 4, 4]
        self.db4_weights = torch.zeros(4, 1, 4, 4)

        # LL (低通-低通)
        self.db4_weights[0, 0, :, :] = torch.outer(db4_low, db4_low)
        # LH (低通-高通)
        self.db4_weights[1, 0, :, :] = torch.outer(db4_low, db4_high)
        # HL (高通-低通)
        self.db4_weights[2, 0, :, :] = torch.outer(db4_high, db4_low)
        # HH (高通-高通)
        self.db4_weights[3, 0, :, :] = torch.outer(db4_high, db4_high)

        # 扩展通道维度并设为不可训练参数
        self.db4_weights = torch.cat([self.db4_weights] * self.channel_in, 0)
        self.db4_weights = nn.Parameter(self.db4_weights)
        self.db4_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            # 逆向变换 (上采样)
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)  # Jacobian调整

            # 重排列输入 [B, C*4, H, W] -> [B, C, 4, H, W]
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])

            # DB4反卷积上采样 (需要padding=2保持尺寸)
            return F.conv_transpose2d(out, self.db4_weights,
                                      stride=2,
                                      padding=1,  # 补偿4x4核的尺寸
                                      groups=self.channel_in)
        else:
            # 正向变换 (下采样)
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            # DB4卷积下采样 (padding=1保持尺寸)
            out = F.conv2d(x, self.db4_weights,
                           stride=2,
                           padding=1,  # 补偿4x4核的尺寸
                           groups=self.channel_in) / 4.0  # 能量归一化

            # 重排列输出 [B, C, 4, H/2, W/2]
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out

    def jacobian(self, x, rev=False):
        return self.last_jac


class Daubechies4Downsampling(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.channel_in = channel_in

        # DB4滤波器系数
        db4_low = torch.tensor([0.48296, 0.83652, 0.22414, -0.12941], dtype=torch.float32)
        db4_high = torch.tensor([-0.12941, -0.22414, 0.83652, -0.48296], dtype=torch.float32)

        # 构建2D滤波器核
        self.db4_weights = torch.zeros(4, 1, 4, 4)
        self.db4_weights[0, 0, :, :] = torch.outer(db4_low, db4_low)  # LL
        self.db4_weights[1, 0, :, :] = torch.outer(db4_low, db4_high)  # LH
        self.db4_weights[2, 0, :, :] = torch.outer(db4_high, db4_low)  # HL
        self.db4_weights[3, 0, :, :] = torch.outer(db4_high, db4_high)  # HH

        # 扩展通道维度
        self.db4_weights = torch.cat([self.db4_weights] * channel_in, 0)
        self.db4_weights = nn.Parameter(self.db4_weights, requires_grad=False)

        # 存储填充尺寸用于逆向变换
        self.pad_H = 0
        self.pad_W = 0

    def forward(self, x, rev=False):
        B, C, H, W = x.shape

        if not rev:
            # 下采样路径 --------------------------------------------------
            self.elements = C * H * W
            self.last_jac = self.elements / 4 * np.log(1 / 16.)

            # 计算需要的填充量
            self.pad_H = (4 - H % 4) % 4
            self.pad_W = (4 - W % 4) % 4

            # 对称填充
            x_padded = F.pad(x, (0, self.pad_W, 0, self.pad_H), mode='reflect')

            # 卷积下采样
            x = F.conv2d(x_padded, self.db4_weights,
                         stride=2,
                         padding=1,  # 补偿4x4核
                         groups=self.channel_in) / 4.0  # 能量归一化

            # 重排列通道
            out_H, out_W = (H + self.pad_H) // 2, (W + self.pad_W) // 2
            x = x.view(B, self.channel_in, 4, out_H, out_W)
            return x.transpose(1, 2).reshape(B, 4 * self.channel_in, out_H, out_W)

        else:
            # 上采样路径 --------------------------------------------------
            self.elements = C * H * W
            self.last_jac = self.elements / 4 * np.log(16.)

            # 重排列输入
            x = x.view(B, self.channel_in, 4, H, W)
            x = x.transpose(1, 2).reshape(B, 4 * self.channel_in, H, W)

            # 反卷积上采样
            out = F.conv_transpose2d(x, self.db4_weights,
                                     stride=2,
                                     padding=1,
                                     output_padding=(H % 2, W % 2),
                                     groups=self.channel_in)

            # 裁剪恢复原始尺寸
            orig_H, orig_W = H * 2 - self.pad_H, W * 2 - self.pad_W
            return out[:, :, :orig_H, :orig_W]

    def jacobian(self, x, rev=False):
        return self.last_jac


class ConvDownsampling(nn.Module):
    def __init__(self, scale):
        super(ConvDownsampling, self).__init__()
        self.scale = scale
        self.scale2 = self.scale ** 2

        self.conv_weights = torch.eye(self.scale2)

        if self.scale == 2: # haar init
            self.conv_weights[0] = torch.Tensor([1./4, 1./4, 1./4, 1./4])
            self.conv_weights[1] = torch.Tensor([1./4, -1./4, 1./4, -1./4])
            self.conv_weights[2] = torch.Tensor([1./4, 1./4, -1./4, -1./4])
            self.conv_weights[3] = torch.Tensor([1./4, -1./4, -1./4, 1./4])
        else:
            self.conv_weights[0] = torch.Tensor([1./(self.scale2)] * (self.scale2))

        self.conv_weights = nn.Parameter(self.conv_weights)

    def forward(self, x, rev=False):
        if not rev:
            # downsample
            # may need improvement
            h = x.shape[2]
            w = x.shape[3]
            wpad = 0
            hpad = 0
            if w % self.scale != 0:
                wpad = self.scale - w % self.scale
            if h % self.scale != 0:
                hpad = self.scale - h % self.scale
            if wpad != 0 or hpad != 0:
                padding = (wpad // 2, wpad - wpad // 2, hpad // 2, hpad - hpad // 2)
                pad = nn.ReplicationPad2d(padding)
                x = pad(x)

            [B, C, H, W] = list(x.size())
            x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(B, C * self.scale2, H // self.scale, W // self.scale)

            # conv
            conv_weights = self.conv_weights.reshape(self.scale2, self.scale2, 1, 1)
            conv_weights = conv_weights.repeat(C, 1, 1, 1)

            out = F.conv2d(x, conv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale2, H // self.scale, W // self.scale)
            out = torch.transpose(out, 1, 2)
            out = out.reshape(B, C * self.scale2, H // self.scale, W // self.scale)

            return out
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.scale2, self.scale2, 1, 1)

            [B, C_, H_, W_] = list(x.size())
            C = C_ // self.scale2
            H = H_ * self.scale
            W = W_ * self.scale

            inv_weights = inv_weights.repeat(C, 1, 1, 1)

            x = x.reshape(B, self.scale2, C, H_, W_)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(B, C_, H_, W_)

            out = F.conv2d(x, inv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale, self.scale, H_, W_)
            out = out.permute(0, 1, 4, 2, 5, 3)
            out = out.reshape(B, C, H, W)

            return out


class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, down_first=False, use_ConvDownsampling=False, down_scale=4):
        super(InvRescaleNet, self).__init__()

        operations = []

        if use_ConvDownsampling:
            down_num = 1
            down_first = True

        current_channel = channel_in
        if down_first:
            for i in range(down_num):
                if use_ConvDownsampling:
                    b = ConvDownsampling(down_scale)
                    current_channel *= down_scale**2
                else:
                    b = HaarDownsampling(current_channel)
                    current_channel *= 4
                operations.append(b)
            for j in range(block_num[0]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
        else:
            for i in range(down_num):
                b = HaarDownsampling(current_channel)
                operations.append(b)
                current_channel *= 4
                for j in range(block_num[i]):
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                    operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out



def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)



class ResidualUnit(nn.Module):
    """Simple residual unit."""
    def __init__(self, emd_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(emd_out, emd_out//2),
            nn.LeakyReLU(0.1, inplace=True),
            conv3x3(emd_out//2, emd_out//2),
            nn.LeakyReLU(0.1, inplace=True),
            conv1x1(emd_out//2, emd_out),
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, rev=False) :
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out
## mlkk
class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, emd_out):
        super().__init__()

        self.conv_a = nn.Sequential(ResidualUnit(emd_out), ResidualUnit(emd_out), ResidualUnit(emd_out))

        self.conv_b = nn.Sequential(
            ResidualUnit(emd_out),
            ResidualUnit(emd_out),
            ResidualUnit(emd_out),
            conv1x1(emd_out, emd_out),
        )

    def forward(self, x, rev=False) :
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class Robust_Module(nn.Module):
    def __init__(self, channel_in=3, mid_ch=64):
        super(Robust_Module, self).__init__()
        self.net = MWCNN()
        # self.head = nn.Conv2d(in_channels=channel_in, out_channels=mid_ch, kernel_size=1)
        #
        # self.Conv_atten1 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  AttentionBlock(mid_ch),
        #                                 )
        #
        # self.Conv_atten2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  AttentionBlock(mid_ch),
        #                                 )
        #
        # self.Conv_atten3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 1),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  ResidualUnit(mid_ch),
        #                                  AttentionBlock(mid_ch),
        #                                  )
        #
        # self.tail = nn.Conv2d(mid_ch, channel_in, 1)

    def forward(self, x_ref, x_fake):

        # identify = x
        # x = self.head(x)
        # x = self.Conv_atten1(x) + x
        # x = self.Conv_atten2(x) + x
        # x = self.Conv_atten3(x) + x
        # out = self.tail(x) + identify
        # xIn = torch.cat((x_ref, x_fake), dim=1)
        x = self.net(x_fake)


        return x

#

# class Conv1x1_expand(nn.Module):
#     def __init__(self, channel_in, channel_out):
#         super(Conv1x1_expand, self).__init__()
#
#         self.conv1 = DenseBlock(channel_in, channel_out)
#         self.conv2 = DenseBlock(channel_out, channel_in)
#
#     def forward(self, x, rev=False):
#         if not rev:
#             out = self.conv1(x)
#         else:
#             out = self.conv2(x)
#         return out


class Conv1x1Trans(nn.Module):
    def __init__(self, channel_in=3):
        super(Conv1x1Trans, self).__init__()

        self.channel_in = channel_in
        self.conv_weights = torch.eye(self.channel_in)
        # if rgb_type == 'RGB':
        #     self.conv_weights[0] = torch.Tensor([0.299, 0.587, 0.114])
        #     self.conv_weights[1] = torch.Tensor([-0.147, -0.289, 0.436])
        #     self.conv_weights[2] = torch.Tensor([0.615, -0.515, -0.100])
        # elif rgb_type == 'BGR':
        #     self.conv_weights[0] = torch.Tensor([0.114, 0.587, 0.299])
        #     self.conv_weights[1] = torch.Tensor([0.436, -0.289, -0.147])
        #     self.conv_weights[2] = torch.Tensor([-0.100, -0.515, 0.615])
        # else:
        #     print("Error! Undefined RGB type!")
        #     exit(1)
        self.conv_weights[0] = torch.Tensor([0.299, 0.587, 0.114])
        self.conv_weights[1] = torch.Tensor([-0.147, -0.289, 0.436])
        self.conv_weights[2] = torch.Tensor([0.615, -0.515, -0.100])

        self.conv_weights = nn.Parameter(self.conv_weights)

        # if not learnable:
        #     self.conv_weights.requires_grad = False

    def forward(self, x, rev=False):
        # print("x IN ", x.shape)
        if not rev:
            conv_weights = self.conv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            out = F.conv2d(x, conv_weights, bias=None, stride=1)
            return out
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            out = F.conv2d(x, inv_weights, bias=None, stride=1)
            return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # prior
        # if group == 1:
        #     self.ln1 = nn.Linear(embed_dim*4, dim)
        #     self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # if prior is not None:
        #     k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
        #     k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
        #     x = (x * k1) + k2

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # # # # # 沿着指定的维度进行平均池化   mlkk  del
        # q = torch.mean(q, dim=2, keepdim=True)
        # k = torch.mean(k, dim=3, keepdim=True)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_conv(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_conv, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        # if prior is not None:
        #     k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
        #     k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
        #     x = (x * k1) + k2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

## refer to Hi-DIff
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_conv(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x

        return x


class Dense_Trans_Block(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32):
        super(Dense_Trans_Block, self).__init__()
        self.global1 = BasicLayer(channel_in, gc, 1)
        self.global2 = BasicLayer(channel_in + gc, gc, 1)
        self.global3 = BasicLayer(channel_in + 2 * gc, gc, 1)
        self.global4 = BasicLayer(channel_in + 3 * gc, gc, 1)
        self.global5 = BasicLayer(channel_in + 4 * gc, channel_out, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.global1(x))
        x2 = self.lrelu(self.global2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.global3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.global4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.global5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class BasicLayer(nn.Module):
    def __init__(self, dim, dim_out, num_heads=2, ffn_expansion_factor=1.66, bias=False, LayerNorm_type='WithBias', num_blocks=1):

        super().__init__()
        # self.project = nn.Conv2d(dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in
             range(num_blocks)])

        self.out = nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # x =self.project(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out(x)
        return x


class Gaussianize(nn.Module):
    def __init__(self, n_channels, out_channel):
        super().__init__()
        # self.net = nn.Sequential(DenseBlock(n_channels, out_channel))
        self.h_mean_s = DenseBlock(n_channels, out_channel, gc=64)
        self.h_scale_s = DenseBlock(n_channels, out_channel, gc=64)                   # computes the parameters of Gaussian
        self.clamp = 1.
        self.affine_eps = 0.000001

    def forward(self, x1, x2, rev=False):
        if not rev:
            # h = self.net(x1)
            m = self.h_mean_s(x1)
            s = self.h_scale_s(x1)
            # m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]          # split along channel dims
            # s = torch.abs(s) + self.affine_eps
            # m, s = h[:, :3, :, :], h[:, 3:, :, :]          # split along channel dims
            # z2 = (x2 - m) / self.e(s)                # center and scale; log prob is computed at the model forward
            z2 = (x2 - m) / self.e(s)                 # center and scale; log prob is computed at the model forward
            return z2
        else:
            z2 = x2
            # h = self.net(x1)
            m = self.h_mean_s(x1)
            s = self.h_scale_s(x1)
            # s = torch.abs(self.h_scale_s(x1)) + self.affine_eps
            # m, s = h[:, 0::2, :, :], h[:, 1::2, :, :]
            # s = torch.abs(s) + self.affine_eps
            x2 = m + z2 * self.e(s)
            # x2 = m + z2 * s
            return x2

    def e(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps
        # return F.softplus(self.clamp * (torch.sigmoid(s) * 2 - 1)) + self.affine_eps


class Conv_Expand(nn.Module):
    def __init__(self, n_channels=3, Gau_channel_scale=1):
        super().__init__()
        self.Gau_channel_scale = Gau_channel_scale

        self.conv1 = Conv1x1Trans(channel_in=n_channels)
        self.conv2 = Conv1x1Trans(channel_in=n_channels)
        if Gau_channel_scale == 2:  ##  最终的 高斯空间通道是x的2倍
            self.conv3 = Conv1x1Trans(channel_in=n_channels)
        elif  Gau_channel_scale == 3:
            self.conv3 = Conv1x1Trans(channel_in=n_channels)
            self.conv4 = Conv1x1Trans(channel_in=n_channels)

    def forward(self, x, rev=False):
        if self.Gau_channel_scale == 2:  ##  2*3 = 6  6+3 = 9
            if not rev:
                input1 = x / 3.0
                input2 = x - x / 3.0 - input1
                input3 = x - (input1 + input2)

                out1 = self.conv1(input1, rev=rev)
                out2 = self.conv2(input2, rev=rev)
                out3 = self.conv3(input3, rev=rev)
                out = torch.cat((out1, out2, out3), dim=1)

            else:
                # x = self.conv3(x, rev=rev)
                out1, out2, out3 = x[:, :3, :, :], x[:, 3:6, :, :], x[:, 6:, :, :]
                out1 = self.conv1(out1, rev=rev)
                out2 = self.conv2(out2, rev=rev)
                out3 = self.conv3(out3, rev=rev)
                out = out1 + out2 + out3
            return out

        elif self.Gau_channel_scale == 3: ##  3*3 = 9， 9+3= 12
            if not rev:
                input1 = x / 4.0
                input2 = x - 2*x / 4.0 - input1
                input3 = x - x / 4.0 - input1 - input2
                input4 = x - (input1 + input2 + input3)

                out1 = self.conv1(input1, rev=rev)
                out2 = self.conv2(input2, rev=rev)
                out3 = self.conv3(input3, rev=rev)
                out4 = self.conv4(input4, rev=rev)
                out = torch.cat((out1, out2, out3, out4), dim=1)

            else:
                # x = self.conv3(x, rev=rev)
                out1, out2, out3, out4 = x[:, :3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :]
                out1 = self.conv1(out1, rev=rev)
                out2 = self.conv2(out2, rev=rev)
                out3 = self.conv3(out3, rev=rev)
                out4 = self.conv3(out4, rev=rev)
                out = out1 + out2 + out3 + out4
            return out


        elif self.Gau_channel_scale == 1:
            if not rev:
                input1 = x/2.0
                input2 = x - input1
                out1 = self.conv1(input1, rev=rev)
                out2 = self.conv2(input2, rev=rev)
                out = torch.cat((out1, out2), dim=1)
                # out = self.conv3(out, rev=rev)
            else:
                # x = self.conv3(x, rev=rev)
                out1, out2 = x[:,:3,:,:], x[:,3:,:,:]
                out1 = self.conv1(out1, rev=rev)
                out2 = self.conv2(out2, rev=rev)
                out = out1 + out2
            return out

class InvComNet(nn.Module):
    def __init__(self, rgb_type, subnet_constructor=None, block_num=[], use_robust=None, Gau_channel_scale=1):
        super(InvComNet, self).__init__()


        self.Wave_type = "Haar"  # Haar db4
        # print("self.Wave_type ?", self.Wave_type)
        self.use_robust = use_robust

        if Gau_channel_scale == 1:
            channel_in = 6
            channel_out = 6
        elif Gau_channel_scale == 2:
            channel_in = 9
            channel_out = 9
        elif Gau_channel_scale == 3:
            channel_in = 12
            channel_out = 12

        operations_gaussian = []

        b = Conv_Expand(n_channels=3, Gau_channel_scale=Gau_channel_scale)
        operations_gaussian.append(b)

        if  self.Wave_type == "Haar":  # Haar
            b = HaarDownsampling(channel_in)
        elif  self.Wave_type == "db4":
            b = Daubechies4Downsampling(channel_in)
        operations_gaussian.append(b)

        for j in range(block_num[0]):
            b = InvBlockExp(subnet_constructor, channel_in*4, channel_out)
            operations_gaussian.append(b)
        if self.Wave_type == "Haar":  # Haar
            b = HaarUping(channel_in)
        elif self.Wave_type == "db4":  # Haar
            b = DB4Uping(channel_in)

        operations_gaussian.append(b)


        # ##  mlkk add for robust  用于约束最后的图像
        # if use_robust:
        #     print(" with Robust Attention \n")
        #     self.Robust_Net = Unet(In_channel=3)
        # else:
        #     print(" without Robust Attention\n")

        self.operations_gaussian = nn.ModuleList(operations_gaussian)
        self.guassianize = Gaussianize(n_channels=3, out_channel=(channel_in-3))

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0
        if not rev:
            for op in self.operations_gaussian:
                x = op.forward(x, rev)
                # print("x shape", x.shape)
                # if cal_jacobian:
                #     jacobian += op.jacobian(out, rev)
            out_compress,  out_gau = x[:,:3,:,:], x[:,3:,:,:]

            z = self.guassianize(x1=out_compress, x2=out_gau, rev=rev)
            out = torch.cat((out_compress, z), dim=1)
        else:
            g, z = out[:, :3, :, :], out[:, 3:, :, :]
            z = self.guassianize(x1=g, x2=z, rev=rev)
            out = torch.cat((g, z), dim=1)

            for op in reversed(self.operations_gaussian):
                out = op.forward(out, rev)
                # if cal_jacobian:
                #     jacobian += op.jacobian(out, rev)
                # else:
                #     return out
            # if self.use_robust:
            #     out_EN = self.Robust_Net(x_LQ, out)  ## 产生的纹理细节和原图进一步产生
            #     return out, out_EN
            # else:
            return out

        if cal_jacobian:
            return out, jacobian
        else:
            return out


