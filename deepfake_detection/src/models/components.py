"""
All sub-modules: HaarDWT2d, StatisticalPooling,
MultiStreamFrequencyExtractor, MultiScaleDWTBranch,
ExpertPhysicsStreamProjector, MultiStreamFusion,
StreamGating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2d(nn.Module):
    def __init__(self):
        super().__init__()
        haar = torch.tensor(
            [[ 1, 1, 1, 1],
             [ 1,-1, 1,-1],
             [ 1, 1,-1,-1],
             [ 1,-1,-1, 1]], dtype=torch.float32) * 0.5
        self.register_buffer('haar', haar.view(4, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        out = F.conv2d(
            x.reshape(B*C, 1, H, W), self.haar, stride=2)
        out = out.reshape(B, C, 4, H//2, W//2)
        return out[:,:,0], out[:,:,1], out[:,:,2], out[:,:,3]


class StatisticalPooling(nn.Module):
    def forward(self, x):
        flat = x.flatten(2)
        mean = flat.mean(2)
        diff = flat - mean.unsqueeze(2)
        var  = (diff**2).mean(2)
        std  = var.clamp(min=1e-6).sqrt()
        skew = (diff**3).mean(2) / std.pow(3)
        kurt = (diff**4).mean(2) / var.clamp(min=1e-6).pow(2)
        skew = skew.clamp(-10., 10.)
        kurt = kurt.clamp(0.,   50.)
        return torch.cat([mean, var, skew, kurt], dim=1)


class SpectralSlopeDetector(nn.Module):
    def __init__(self, out_features=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.GELU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(out_features//2),
            nn.Flatten(),
            nn.Linear(32*(out_features//2), out_features),
        )
    def forward(self, psd):
        return self.net(psd.unsqueeze(1))


class PatchPhysicsTokenizer(nn.Module):
    def __init__(self, patch_size=16, d_model=256):
        super().__init__()
        self.P    = patch_size
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x):
        B, C, H, W = x.shape
        P      = min(self.P, H, W)
        nH, nW = H // P, W // P
        if nH == 0 or nW == 0:
            d = self.proj[0].weight.shape[0]
            return torch.zeros(B, 1, d, device=x.device)
        gray    = x.mean(1, keepdim=True)
        patches = gray.unfold(2, P, P).unfold(3, P, P)
        patches = patches.reshape(B, nH*nW, P*P)
        coeffs  = torch.fft.rfft(patches, dim=-1).abs()
        mean    = coeffs.mean(-1)
        diff    = coeffs - mean.unsqueeze(-1)
        var     = (diff**2).mean(-1)
        std     = var.clamp(min=1e-6).sqrt()
        skew    = (diff**3).mean(-1) / std.pow(3)
        kurt    = (diff**4).mean(-1) / var.clamp(min=1e-6).pow(2)
        skew    = skew.clamp(-10., 10.)
        kurt    = kurt.clamp(0.,   100.)
        return self.proj(
            torch.stack([mean, var, skew, kurt], dim=-1))


class SubBandProcessor(nn.Module):
    def __init__(self, in_channels=3, d_model=256, spatial_out=7):
        super().__init__()
        self.spatial_out = spatial_out
        self.stat_pool   = StatisticalPooling()
        self.conv_stream = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*4, 3, 1, 1,
                      groups=in_channels),
            nn.Conv2d(in_channels*4, d_model, 1),
            nn.BatchNorm2d(d_model), nn.GELU(),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(4*in_channels, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
        )

    def forward(self, band):
        conv_tokens = F.adaptive_avg_pool2d(
            self.conv_stream(band),
            self.spatial_out).flatten(2).transpose(1, 2)
        stat_token  = self.stat_proj(
            self.stat_pool(band)).unsqueeze(1)
        return conv_tokens, stat_token


class MultiStreamFrequencyExtractor(nn.Module):
    def __init__(self, d_model=256, img_size=224):
        super().__init__()
        patch_size      = 16 if img_size >= 64 else 4
        self.dwt        = HaarDWT2d()
        self.stat_pool  = StatisticalPooling()
        self.patch_phys = PatchPhysicsTokenizer(patch_size, d_model)
        self.slope_det  = SpectralSlopeDetector(out_features=16)
        self.hh_conv    = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(64, d_model, 1),
        )
        self.hh_weight  = nn.Parameter(torch.ones(1))
        self.stat_proj  = nn.Sequential(
            nn.Linear(4*3, d_model),
            nn.LayerNorm(d_model), nn.GELU())
        self.slope_proj = nn.Sequential(
            nn.Linear(16, d_model),
            nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x, psd_curve):
        LL, LH, HL, HH = self.dwt(x)
        hh_tokens    = self.hh_conv(
            HH * self.hh_weight).flatten(2).transpose(1, 2)
        patch_tokens = self.patch_phys(x)
        stats        = (self.stat_pool(HH)
                        + self.stat_pool(LH)
                        + self.stat_pool(HL))
        stat_token   = self.stat_proj(stats).unsqueeze(1)
        slope_token  = self.slope_proj(
            self.slope_det(psd_curve)).unsqueeze(1)
        return hh_tokens, patch_tokens, stat_token, slope_token


class MultiScaleDWTBranch(nn.Module):
    def __init__(self, in_channels=3, d_model=256,
                 num_levels=3, spatial_out=7):
        super().__init__()
        self.num_levels = num_levels
        self.dwt        = HaarDWT2d()
        self.processors = nn.ModuleDict({
            f'L{l}_{b}': SubBandProcessor(
                in_channels, d_model, spatial_out)
            for l in range(1, num_levels+1)
            for b in ['LH', 'HL', 'HH']
        })
        w = torch.tensor([[1.0, 1.0, 1.5]] * num_levels)
        self.band_weights = nn.Parameter(w)
        self.cross_scale  = nn.Sequential(
            nn.Linear(num_levels*3*d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        all_spatial, all_stat = [], []
        ll = x
        for lv in range(1, self.num_levels + 1):
            if ll.shape[-1] < 2 or ll.shape[-2] < 2:
                break
            LL, LH, HL, HH = self.dwt(ll)
            for i, (name, band) in enumerate(
                    [('LH', LH), ('HL', HL), ('HH', HH)]):
                w  = F.softplus(self.band_weights[lv-1, i])
                ct, st = self.processors[f'L{lv}_{name}'](
                    band * w)
                all_spatial.append(ct)
                all_stat.append(st)
            ll = LL
        spatial_tokens    = torch.cat(all_spatial, dim=1)
        stat_tokens       = torch.cat(all_stat,    dim=1)
        cross_scale_token = self.cross_scale(
            stat_tokens.flatten(1)).unsqueeze(1)
        return spatial_tokens, stat_tokens, cross_scale_token


class StreamGating(nn.Module):
    def __init__(self, num_streams, d_model, min_gate=0.1):
        super().__init__()
        self.num_streams = num_streams
        self.min_gate    = min_gate
        self.norms       = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_streams)])
        self.gate_net    = nn.Sequential(
            nn.Linear(num_streams * d_model, d_model),
            nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, num_streams),
        )

    def forward(self, streams):
        summaries = [
            self.norms[i](s).mean(1)
            for i, s in enumerate(streams)]
        gates = F.softmax(
            self.gate_net(torch.cat(summaries, -1)), dim=-1)
        gates = self.min_gate + (1 - self.min_gate) * gates
        gated = [s * gates[:, i:i+1, None]
                 for i, s in enumerate(streams)]
        return gated, gates


class ExpertPhysicsStreamProjector(nn.Module):
    GROUPS = {
        'psd'     : (0,  25),
        'prnu_snr': (25, 28),
        'phase'   : (28, 39),
        'forensic': (39, 52),
    }

    def __init__(self, d_model=256):
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(end - start, d_model),
                nn.LayerNorm(d_model), nn.GELU(),
            )
            for name, (start, end) in self.GROUPS.items()
        })

    def forward(self, phys):
        return {
            name: self.projectors[name](
                phys[:, start:end]).unsqueeze(1)
            for name, (start, end) in self.GROUPS.items()
        }


class MultiStreamFusion(nn.Module):
    STREAM_NAMES = [
        'hh_conv_p1', 'patch_moments_p1',
        'global_stat_p1', 'spectral_slope_p1',
        'dwt_spatial_p2', 'dwt_stat_p2', 'cross_scale_p2',
        'expert_psd', 'expert_prnu_snr',
        'expert_phase', 'expert_forensic',
    ]
    NUM_STREAMS = 11

    def __init__(self, d_model=256, min_gate=0.1):
        super().__init__()
        self.gating = StreamGating(
            self.NUM_STREAMS, d_model, min_gate)

    def forward(self, hh_p1, patch_p1, stat_p1, slope_p1,
                sp_p2, stat_p2, cs_p2,
                ex_psd, ex_prnu_snr, ex_phase, ex_forensic,
                cls_token):
        streams = [
            hh_p1, patch_p1, stat_p1, slope_p1,
            sp_p2,  stat_p2, cs_p2,
            ex_psd, ex_prnu_snr, ex_phase, ex_forensic,
        ]
        gated, gates = self.gating(streams)
        return torch.cat([cls_token] + gated, dim=1), gates
