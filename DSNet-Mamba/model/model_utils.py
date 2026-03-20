import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vmamba_medical import VSSBlock
except ImportError:
    from vmamba_medical import VSSBlock

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes * upscale_factor ** 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes * upscale_factor ** 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                                  dilation=dilation, padding=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class MFACB(nn.Module):
    """Multi-scale Fusion Atrous Convolution Block."""
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2, 2, 2]):
        super(MFACB, self).__init__()
        assert block_num > 1
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes * 3, out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1


class SPASPP(nn.Module):
    """Serial-Parallel Atrous Spatial Pyramid Pooling (original DSNet version)."""
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1,
                 dilation=[6, 12, 18, 24]):
        super(SPASPP, self).__init__()
        assert block_num > 1
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[3]))
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes * 5, out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False)
        )
        self.process3 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        out2 = self.pooling(x)
        for idx in range(4):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out, out2), dim=1)
        return self.process3(self.process2(out) + out1)


class SPASPP_VSS(nn.Module):
    """
    SPASPP-M: Mamba-Enhanced Serial-Parallel Spatial Pyramid Pooling.
    Replaces serial atrous convolutions with sequential VSSBlocks.
    """
    def __init__(self, in_planes, inter_planes, out_planes, depth: int = 4,
                 drop_path: float = 0.0, d_state: int = 16):
        super().__init__()
        assert 1 <= depth <= 4, "depth must be between 1 and 4"
        self.depth = depth
        self.vss_in_proj = nn.Conv2d(inter_planes, 192, kernel_size=1, bias=False) \
            if inter_planes != 192 else nn.Identity()
        self.vss_out_proj = nn.Conv2d(192, inter_planes, kernel_size=1, bias=False) \
            if inter_planes != 192 else nn.Identity()
        self.proj = nn.Identity() if in_planes == inter_planes else Conv1X1(in_planes, inter_planes)
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=192, drop_path=drop_path, d_state=d_state)
            for _ in range(depth)
        ])
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes * (depth + 1), out_planes, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )
        self.process3 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        out1 = self.process1(x)
        out2 = self.pooling(x)
        out = self.proj(x)
        out_list = []
        for blk in self.blocks:
            out = self.vss_in_proj(out)
            out = blk(out)
            out = self.vss_out_proj(out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out, out2), dim=1)
        return self.process3(self.process2(out) + out1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmentheadCARAFE(nn.Module):
    def __init__(self, interplanes, outplanes, scale_factor=8, cp_rate=4):
        super(segmentheadCARAFE, self).__init__()
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.carafe = CARAFE(inC=interplanes, outC=outplanes, cp_rate=cp_rate, up_factor=scale_factor)
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.carafe(self.relu(self.bn2(x)))
        return out


class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=algc)
        return out


class segmenthead_c(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_c, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=algc)
        return out


class segmenthead_drop(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_drop, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout(0.5)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.drop(self.relu(self.bn2(x)))
        out = self.conv2(x)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=algc)
        return out


class AFF(nn.Module):
    """Attentional Feature Fusion."""
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Muti_AFF(nn.Module):
    """
    MSAF-M: Mamba-Enhanced Multi-Scale Attention Fusion.
    Computes pixel-wise fusion weights using Pixel Attention,
    Mamba Region Attention (MRA), and Global Attention branches.
    """
    def __init__(self, channels=64, r=4, use_mamba_region: bool = True,
                 mamba_region_size: int = 16, mamba_drop_path: float = 0.0,
                 mamba_d_state: int = 16, mamba_add_to_multiscale: bool = False):
        super(Muti_AFF, self).__init__()
        inter_channels = int(channels // r)

        # Pixel Attention branch
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Global Attention branch
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.mamba_region_size = mamba_region_size

        # Mamba Region Attention (MRA) branch
        self.mamba_in_proj  = nn.Conv2d(channels, 192, kernel_size=1, bias=False)
        self.mamba_out_proj = nn.Conv2d(192, channels, kernel_size=1, bias=False)
        self.mamba_region   = VSSBlock(
            hidden_dim=192,
            drop_path=mamba_drop_path,
            d_state=mamba_d_state,
        )

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        mr = F.adaptive_avg_pool2d(xa, (self.mamba_region_size, self.mamba_region_size))
        mr = self.mamba_in_proj(mr)
        mr = self.mamba_region(mr)
        mr = self.mamba_out_proj(mr)
        mr = F.interpolate(mr, size=[h, w], mode='nearest')
        xlg = xl + xg + mr
        wei = self.sigmoid(xlg)
        return 2 * x * wei + 2 * residual * (1 - wei)


class MSAF_small(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSAF_small, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]
        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        xg = self.global_att(xa)
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        xlg = xl + xg + c1 + c2
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MSA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSA, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        xa = x
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')
        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei
        return xo


class MSA_small(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSA_small, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        xa = x
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        xg = self.global_att(xa)
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        xlg = xl + xg + c1 + c2
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei
        return xo


class iAFF(nn.Module):
    """Iterative Attentional Feature Fusion."""
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18, 24], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                       size=[height, width], mode='bilinear') + x_list[0])))
        x_list.append(self.process2((F.interpolate(self.scale2(x),
                       size=[height, width], mode='bilinear') + x_list[1])))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                       size=[height, width], mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                       size=[height, width], mode='bilinear') + x_list[3])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2, cp_rate=4, ifBN=False):
        super(CARAFE, self).__init__()
        self.ifBN = ifBN
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // cp_rate, 1)
        self.encoder = nn.Conv2d(inC // cp_rate, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)
        self.bn = nn.BatchNorm2d(outC)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()
        kernel_tensor = self.down(in_tensor)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)
        in_tensor = F.pad(in_tensor,
                          pad=(self.kernel_size // 2, self.kernel_size // 2,
                               self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)
        out_tensor = torch.matmul(in_tensor, kernel_tensor)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        if self.ifBN:
            out_tensor = self.bn(out_tensor)
        return out_tensor