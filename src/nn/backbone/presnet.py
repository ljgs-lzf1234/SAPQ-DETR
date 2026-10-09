"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .common import get_activation, FrozenBatchNorm2d

from ...core import register


__all__ = ['PResNet']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SAPRConvNormLayer(nn.Module):
    """Partial-gated lightweight 3x3 replacement used only in high-level blocks."""

    def __init__(
        self,
        channels,
        partial_ratio=0.25,
        pw_groups=4,
        gate_reduction=16,
        gate_scale=0.25,
    ):
        super().__init__()
        channels = int(channels)
        partial_channels = max(1, int(round(channels * float(partial_ratio))))
        partial_channels = min(partial_channels, channels)
        pw_groups = max(1, int(pw_groups))
        while channels % pw_groups != 0 and pw_groups > 1:
            pw_groups -= 1

        hidden = max(8, channels // int(gate_reduction))
        self.partial_channels = partial_channels
        self.gate_scale = float(gate_scale)

        self.rep3 = ConvNormLayer(partial_channels, partial_channels, 3, 1, act=None)
        self.rep1 = ConvNormLayer(partial_channels, partial_channels, 1, 1, padding=0, act=None)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, groups=pw_groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.gate[-1].weight, 0.)
        nn.init.constant_(self.gate[-1].bias, 0.)

    def forward(self, x):
        xa = x[:, :self.partial_channels]
        xb = x[:, self.partial_channels:]
        ya = self.rep3(xa) + self.rep1(xa) + xa
        y = ya if xb.numel() == 0 else torch.cat([ya, xb], dim=1)
        y = self.norm(self.mix(y))
        gate = 1. + self.gate_scale * torch.tanh(self.gate(y))
        return y * gate


class ShallowDetailResidual(nn.Module):
    """Foreground-aware high-pass compensation for C2/C3 small-object detail."""

    def __init__(
        self,
        channels,
        eta_init=0.04,
        eta_max=0.12,
        gate_reduction=16,
        use_fg_bg_smoothing=False,
        bg_smooth_beta_init=0.03,
        bg_smooth_beta_max=0.10,
    ):
        super().__init__()
        channels = int(channels)
        self.eta_max = float(eta_max)
        self.use_fg_bg_smoothing = bool(use_fg_bg_smoothing)
        eta_init = min(max(float(eta_init), 1e-6), self.eta_max - 1e-6)
        eta_prob = eta_init / self.eta_max
        self.eta_logit = nn.Parameter(torch.tensor(math.log(eta_prob / (1. - eta_prob))))

        self.bg_smooth_beta_max = float(bg_smooth_beta_max)
        bg_smooth_beta_init = min(
            max(float(bg_smooth_beta_init), 1e-6),
            self.bg_smooth_beta_max - 1e-6,
        )
        beta_prob = bg_smooth_beta_init / self.bg_smooth_beta_max
        self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_prob / (1. - beta_prob))))

        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        hidden = max(8, channels // int(gate_reduction))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].weight, 0.)
        nn.init.constant_(self.gate[-2].bias, 0.)

        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.fg_gate[-2].weight, 0.)
        nn.init.constant_(self.fg_gate[-2].bias, 0.)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        detail = self.norm(self.dw(high))
        eta = self.eta_max * torch.sigmoid(self.eta_logit.to(dtype=x.dtype))
        if not self.use_fg_bg_smoothing:
            return x + eta * self.gate(high.abs()) * detail

        high_abs = high.abs()
        spatial_stat = torch.cat(
            [
                high_abs.mean(dim=1, keepdim=True),
                high_abs.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        fg = self.fg_gate(spatial_stat)
        beta = self.bg_smooth_beta_max * torch.sigmoid(self.beta_logit.to(dtype=x.dtype))
        return x + eta * fg * self.gate(high_abs) * detail - beta * (1. - fg) * high


class LowFrequencySemanticRecovery(nn.Module):
    """Lightweight low-frequency recovery for compressed C5 semantics."""

    def __init__(
        self,
        channels,
        gate_init=0.05,
        gate_max=0.20,
        pw_groups=8,
        gate_reduction=16,
    ):
        super().__init__()
        channels = int(channels)
        pw_groups = max(1, int(pw_groups))
        while channels % pw_groups != 0 and pw_groups > 1:
            pw_groups -= 1

        self.gate_max = float(gate_max)
        gate_init = min(max(float(gate_init), 1e-6), self.gate_max - 1e-6)
        gate_prob = gate_init / self.gate_max
        self.recovery_logit = nn.Parameter(torch.tensor(math.log(gate_prob / (1. - gate_prob))))

        self.low_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.low_norm = nn.BatchNorm2d(channels)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, groups=pw_groups, bias=False)
        self.mix_norm = nn.BatchNorm2d(channels)
        hidden = max(8, channels // int(gate_reduction))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].weight, 0.)
        nn.init.constant_(self.gate[-2].bias, 0.)

    @staticmethod
    def _haar_low(x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return 0.5 * (x00 + x01 + x10 + x11), (pad_h, pad_w, h, w)

    @staticmethod
    def _haar_restore_low(ll, meta):
        pad_h, pad_w, h, w = meta
        out = ll.new_zeros(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2)
        value = 0.5 * ll
        out[:, :, 0::2, 0::2] = value
        out[:, :, 0::2, 1::2] = value
        out[:, :, 1::2, 0::2] = value
        out[:, :, 1::2, 1::2] = value
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out

    def forward(self, x):
        ll, meta = self._haar_low(x)
        rec = self.low_norm(self.low_dw(ll))
        rec = self.mix_norm(self.mix(rec))
        rec = self._haar_restore_low(rec, meta)
        gamma = self.gate_max * torch.sigmoid(self.recovery_logit.to(dtype=x.dtype))
        return x + gamma * self.gate(x) * rec


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        ch_in,
        ch_out,
        stride,
        shortcut,
        act='relu',
        variant='b',
        use_sapr=False,
        sapr_partial_ratio=0.25,
        sapr_pw_groups=4,
        sapr_gate_reduction=16,
        sapr_gate_scale=0.25,
    ):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        if use_sapr:
            self.branch2b = SAPRConvNormLayer(
                ch_out,
                partial_ratio=sapr_partial_ratio,
                pw_groups=sapr_pw_groups,
                gate_reduction=sapr_gate_reduction,
                gate_scale=sapr_gate_scale,
            )
        else:
            self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(
        self,
        block,
        ch_in,
        ch_out,
        count,
        stage_num,
        act='relu',
        variant='b',
        sapr_block_indices=None,
        sapr_partial_ratio=0.25,
        sapr_pw_groups=4,
        sapr_gate_reduction=16,
        sapr_gate_scale=0.25,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        sapr_indices = set()
        if sapr_block_indices is not None:
            for idx in sapr_block_indices:
                idx = int(idx)
                sapr_indices.add(count + idx if idx < 0 else idx)

        for i in range(count):
            block_kwargs = dict(
                stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=False if i == 0 else True,
                variant=variant,
                act=act,
            )
            if block is BasicBlock:
                block_kwargs.update(
                    use_sapr=i in sapr_indices,
                    sapr_partial_ratio=sapr_partial_ratio,
                    sapr_pw_groups=sapr_pw_groups,
                    sapr_gate_reduction=sapr_gate_reduction,
                    sapr_gate_scale=sapr_gate_scale,
                )
            self.blocks.append(block(ch_in, ch_out, **block_kwargs))

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register()
class PResNet(nn.Module):
    def __init__(
        self,
        depth,
        variant='d',
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
        use_sapr_backbone=False,
        sapr_detail_stages=[0, 1],
        sapr_detail_eta_init=0.04,
        sapr_detail_eta_max=0.12,
        sapr_detail_gate_reduction=16,
        sapr_use_fg_bg_smoothing=False,
        sapr_bg_smooth_beta_init=0.03,
        sapr_bg_smooth_beta_max=0.10,
        sapr_replace_stages=[3],
        sapr_replace_blocks=[-1],
        sapr_partial_ratio=0.25,
        sapr_pw_groups=4,
        sapr_gate_reduction=16,
        sapr_gate_scale=0.25,
        sapr_use_c5_lsr=False,
        sapr_lsr_stages=[3],
        sapr_lsr_gate_init=0.05,
        sapr_lsr_gate_max=0.20,
        sapr_lsr_pw_groups=8,
        sapr_lsr_gate_reduction=16):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        self.use_sapr_backbone = bool(use_sapr_backbone)
        sapr_replace_stages = set(int(i) for i in sapr_replace_stages)
        for i in range(num_stages):
            stage_num = i + 2
            sapr_indices = sapr_replace_blocks if self.use_sapr_backbone and i in sapr_replace_stages else None
            self.res_layers.append(
                Blocks(
                    block,
                    ch_in,
                    ch_out_list[i],
                    block_nums[i],
                    stage_num,
                    act=act,
                    variant=variant,
                    sapr_block_indices=sapr_indices,
                    sapr_partial_ratio=sapr_partial_ratio,
                    sapr_pw_groups=sapr_pw_groups,
                    sapr_gate_reduction=sapr_gate_reduction,
                    sapr_gate_scale=sapr_gate_scale,
                )
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        sapr_detail_stages = set(int(i) for i in sapr_detail_stages)
        self.sapr_detail_blocks = nn.ModuleList()
        for i in range(num_stages):
            if self.use_sapr_backbone and i in sapr_detail_stages:
                self.sapr_detail_blocks.append(
                    ShallowDetailResidual(
                        channels=_out_channels[i],
                        eta_init=sapr_detail_eta_init,
                        eta_max=sapr_detail_eta_max,
                        gate_reduction=sapr_detail_gate_reduction,
                        use_fg_bg_smoothing=sapr_use_fg_bg_smoothing,
                        bg_smooth_beta_init=sapr_bg_smooth_beta_init,
                        bg_smooth_beta_max=sapr_bg_smooth_beta_max,
                    )
                )
            else:
                self.sapr_detail_blocks.append(nn.Identity())

        sapr_lsr_stages = set(int(i) for i in sapr_lsr_stages)
        self.sapr_lsr_blocks = nn.ModuleList()
        for i in range(num_stages):
            if self.use_sapr_backbone and bool(sapr_use_c5_lsr) and i in sapr_lsr_stages:
                self.sapr_lsr_blocks.append(
                    LowFrequencySemanticRecovery(
                        channels=_out_channels[i],
                        gate_init=sapr_lsr_gate_init,
                        gate_max=sapr_lsr_gate_max,
                        pw_groups=sapr_lsr_pw_groups,
                        gate_reduction=sapr_lsr_gate_reduction,
                    )
                )
            else:
                self.sapr_lsr_blocks.append(nn.Identity())

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            incompatible = self.load_state_dict(state, strict=not self.use_sapr_backbone)
            if self.use_sapr_backbone:
                missing = len(incompatible.missing_keys)
                unexpected = len(incompatible.unexpected_keys)
                print(f'Load PResNet{depth} state_dict with SAPR ({missing} missing, {unexpected} unexpected)')
            else:
                print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            x = self.sapr_detail_blocks[idx](x)
            x = self.sapr_lsr_blocks[idx](x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
