import warnings

warnings.filterwarnings("ignore")

import math
import torch
from torch import nn
from torch.nn import functional as F

from .module import modules
from .module import attentions
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from .module.commons import init_weights
from .module.mrte_model import MRTE
from .module.quantize import ResidualVectorQuantizer

from ..G2P import Symbols


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        is_bias=False,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    
    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        symbols = Symbols.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.y_overlap = None
    
    def infer(self, y, text, ge, speed, stream_mode=False, valid_start_idx=None, overlap_len=None, slice_indices=None):
        y_mask = torch.ones((1, 1, y.size(2)), dtype=y.dtype, device=y.device)
        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.ones((1, 1, text.size(1)), dtype=text.dtype, device=text.device)
        text = self.text_embedding(text)
        text = text.transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)

        y = self.mrte(y, y_mask, text, text_mask, ge, slice_indices)
        y = self.encoder2(y * y_mask, y_mask)
        
        if stream_mode:
            y = y[:, :, valid_start_idx:]
            y_mask = y_mask[:, :, valid_start_idx:]
            alpha = torch.linspace(0, 1, overlap_len, dtype=y.dtype, device=y.device).view(1, 1, -1)
            if not self.y_overlap is None:
                y[:, :, :overlap_len] = self.y_overlap*(1-alpha) + y[:, :, :overlap_len]*alpha
            self.y_overlap = y[:, :, -overlap_len:]
        
        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, y_mask


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        compile_mode,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.samples_per_frame = math.prod(self.upsample_rates)
        self.semantic_frame_rate = semantic_frame_rate
        
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768

        self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        self.sv_emb = nn.Linear(20480, gin_channels)
        self.ge_to512 = nn.Linear(gin_channels, 512)
        self.prelu = nn.PReLU(num_parameters=gin_channels)

        if compile_mode == "default-optimized" or compile_mode == "max-optimized":
            self.flow.forward = torch.compile(self.flow.forward, mode="default", dynamic=True, fullgraph=True)
            self.dec.forward = torch.compile(self.dec.forward, mode="default", dynamic=True, fullgraph=True)
    
    def get_ge(self, refer, sv_emb):
        refer_mask = torch.ones((1, 1, refer.size(2)), dtype=refer.dtype, device=refer.device)
        ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        sv_emb = self.sv_emb(sv_emb)
        ge += sv_emb.unsqueeze(-1)
        ge = self.prelu(ge)

        return ge
    
    @torch.inference_mode()
    def decode(self, codes, text, ge, noise_scale=0.5, speed=1, stream_mode=False, valid_start_idx=None, overlap_len=None, slice_indices=None):
        quantized = self.quantizer.decode(codes)
        quantized = F.interpolate(quantized, size=quantized.shape[-1] * 2, mode="nearest")
        if ge.shape[-1] != 1: ge = F.interpolate(ge, size=ge.shape[-1] * 2, mode="nearest")

        m_p, logs_p, y_mask = self.enc_p.infer(
            quantized,
            text,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1),
            speed,
            stream_mode,
            valid_start_idx,
            overlap_len,
            slice_indices,
        )

        if speed != 1 and ge.shape[-1] != 1: ge = F.interpolate(ge, size=m_p.shape[-1], mode="nearest")

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, ge)
        o = self.dec(z * y_mask, g=ge)
        
        attn = self.enc_p.mrte.cross_attention.attn
            
        return o, attn[0, ...]

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)