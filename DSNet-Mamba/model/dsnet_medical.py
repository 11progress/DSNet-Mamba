import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from .model_utils import (BasicBlock, Bottleneck, segmenthead, AFF, ASPP,
                               segmentheadCARAFE, iAFF, segmenthead_drop, Muti_AFF,
                               segmenthead_c, DUC, SPASPP, SPASPP_VSS, MFACB, VSSBlock)
except ImportError:
    from model_utils import (BasicBlock, Bottleneck, segmenthead, AFF, ASPP,
                              segmentheadCARAFE, iAFF, segmenthead_drop, Muti_AFF,
                              segmenthead_c, DUC, SPASPP, SPASPP_VSS, MFACB, VSSBlock)

bn_mom = 0.1


class BatchNorm2d(nn.InstanceNorm2d):
    """
    InstanceNorm2d used in place of BatchNorm2d for better performance
    under small batch sizes in medical image segmentation.
    """
    def __init__(self, num_features, momentum=bn_mom):
        super().__init__(num_features, affine=True, track_running_stats=False)


class DSNetMedical(nn.Module):
    """
    DSNet-Mamba: A lightweight Mamba-enhanced network for medical image segmentation.

    Two design modifications over DSNet:
      1. MSAF-M: Mamba Region Attention (MRA) replaces multi-scale region pooling in MSAF.
      2. SPASPP-M: Four sequential VSSBlocks replace serial atrous convolutions in SPASPP.
    """

    def __init__(
        self,
        num_classes=9,
        input_channels=1,
        model_name='s128',
        dsnet_pretrained_path=None,
        vmunet_pretrained_path=None,
    ):
        super(DSNetMedical, self).__init__()

        m = 2
        n = 2
        planes = 32 if 's' in model_name else 64
        name = model_name

        self.name = name
        self.num_classes = num_classes

        # Shared encoder stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)

        # Context Branch: MFACB layers (atrous convolutions)
        self.layer3 = nn.Sequential(
            MFACB(planes * 2, planes * 2, planes * 4, dilation=[2, 2, 2]),
            MFACB(planes * 4, planes * 4, planes * 4, dilation=[2, 2, 2]),
            MFACB(planes * 4, planes * 4, planes * 4, dilation=[3, 3, 3]),
        )
        self.layer4 = nn.Sequential(
            MFACB(planes * 4, planes * 4, planes * 8, dilation=[3, 3, 3]),
            MFACB(planes * 8, planes * 8, planes * 8, dilation=[5, 5, 5]),
        )

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 4, 1, stride=1, dilation=5)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.compression5 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        # Spatial Branch: BasicBlocks
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 4, n)
        self.layer4_ = self._make_layer(BasicBlock, planes * 4, planes * 4, n)
        self.layer5_ = self._make_layer(Bottleneck, planes * 4, planes * 2, 1)

        # MSAF-M: Mamba-enhanced fusion modules
        self.aff1 = Muti_AFF(
            channels=planes * 4,
            use_mamba_region=True,
            mamba_region_size=16,
            mamba_drop_path=0.0,
            mamba_d_state=16,
            mamba_add_to_multiscale=False,
        )
        self.aff2 = Muti_AFF(
            channels=planes * 4,
            use_mamba_region=True,
            mamba_region_size=16,
            mamba_drop_path=0.0,
            mamba_d_state=16,
            mamba_add_to_multiscale=False,
        )
        self.aff3 = Muti_AFF(
            channels=planes * 4,
            use_mamba_region=True,
            mamba_region_size=16,
            mamba_drop_path=0.0,
            mamba_d_state=16,
            mamba_add_to_multiscale=False,
        )

        # SPASPP-M: Mamba-enhanced context extraction module (4 VSSBlocks)
        self.spp = SPASPP_VSS(
            planes * 4, planes * 4, planes * 4,
            depth=4,
            drop_path=0.0,
            d_state=16,
        )

        self.layer1_a = self._make_layer(BasicBlock, planes, planes, 1)
        self.up8 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.lastlayer = segmenthead_c(planes * 5, planes * 4, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if dsnet_pretrained_path:
            self.load_dsnet_pretrained(dsnet_pretrained_path)
        if vmunet_pretrained_path:
            self.load_vmunet_pretrained(vmunet_pretrained_path)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == blocks - 1)
            layers.append(block(inplanes, planes, stride=1, no_relu=no_relu, dilation=dilation))
        return nn.Sequential(*layers)

    def load_dsnet_pretrained(self, pretrained_path):
        """Load DSNet ImageNet pretrained weights (excluding VSSBlock parts)."""
        if not os.path.exists(pretrained_path):
            print(f"Warning: DSNet pretrained path not found: {pretrained_path}")
            return
        print(f"Loading DSNet pretrained weights from: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        if any('module.' in k for k in pretrained_state.keys()):
            pretrained_state = {k[7:]: v for k, v in pretrained_state.items()}
        model_dict = self.state_dict()
        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
            and not any(x in k for x in ['layer3.', 'layer4.', 'vss_block'])
        }
        model_dict.update(filtered_state)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(filtered_state)} DSNet pretrained parameters")

    def load_vmunet_pretrained(self, pretrained_path):
        """Load VMamba-Small pretrained weights for VSSBlock modules."""
        if not os.path.exists(pretrained_path):
            print(f"Warning: VMamba pretrained path not found: {pretrained_path}")
            return
        print(f"Loading VMamba pretrained weights from: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'model' in pretrained_state:
            pretrained_state = pretrained_state['model']
        model_dict = self.state_dict()
        vss_state = {}
        block_map = {'0': 'vss_block1', '1': 'vss_block2'}

        for k, v in pretrained_state.items():
            if k.startswith('layers.1.blocks.'):
                parts = k.split('.')
                if len(parts) < 5:
                    continue
                block_idx = parts[3]
                suffix = '.'.join(parts[4:])
                if block_idx in block_map:
                    new_key = f'layer3.{block_map[block_idx]}.{suffix}'
                    if new_key in model_dict and model_dict[new_key].shape == v.shape:
                        vss_state[new_key] = v
                if block_idx == '0':
                    for aff_name in ['aff1', 'aff2', 'aff3']:
                        new_key = f'{aff_name}.mamba_region.{suffix}'
                        if new_key in model_dict and model_dict[new_key].shape == v.shape:
                            vss_state[new_key] = v
                if block_idx in ['0', '1', '2', '3']:
                    new_key = f'spp.blocks.{block_idx}.{suffix}'
                    if new_key in model_dict and model_dict[new_key].shape == v.shape:
                        vss_state[new_key] = v
            elif k.startswith('layers.2.blocks.'):
                parts = k.split('.')
                if len(parts) < 5:
                    continue
                block_idx = parts[3]
                suffix = '.'.join(parts[4:])
                if block_idx not in block_map:
                    continue
                new_key = f'layer4.{block_map[block_idx]}.{suffix}'
                if new_key in model_dict and model_dict[new_key].shape == v.shape:
                    vss_state[new_key] = v

        if vss_state:
            model_dict.update(vss_state)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(vss_state)} VMamba pretrained parameters")
        else:
            print("Warning: No VMamba parameters matched")

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        height_output = x.shape[-2]
        width_output  = x.shape[-1]

        x = self.conv1(x)
        x = self.layer1(x)
        x_a = self.layer1_a(x)
        x = self.relu(self.layer2(self.relu(x)))

        x_ = self.layer3_(x)
        x  = self.layer3(x)
        x_ = self.aff1(x_, self.compression3(x))

        x  = self.layer4(x)
        x_ = self.layer4_(self.relu(x_))
        x_ = self.aff2(x_, self.compression4(x))

        x_ = self.layer5_(self.relu(x_))
        x  = self.layer5(x)
        x  = self.relu(x)
        x_ = self.aff3(x_, self.compression5(x))

        x_ = self.relu(x_)
        x_ = self.spp(x_)
        x_ = self.up8(x_)
        x_ = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=False)

        x_ = torch.cat((x_, x_a), dim=1)
        x_ = self.lastlayer(x_)
        x_ = F.interpolate(x_, size=[height_output, width_output],
                           mode='bilinear', align_corners=False)
        return x_


def get_dsnet_mamba(num_classes=9, model_name='s128',
                    dsnet_pretrained_path=None, vmunet_pretrained_path=None):
    """Build DSNet-Mamba model."""
    model = DSNetMedical(
        num_classes=num_classes,
        model_name=model_name,
        dsnet_pretrained_path=dsnet_pretrained_path,
        vmunet_pretrained_path=vmunet_pretrained_path,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"DSNet-Mamba | Params: {total / 1e6:.2f}M")
    return model