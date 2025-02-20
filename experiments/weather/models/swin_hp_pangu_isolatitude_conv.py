"""This implementation of the HEAL-SWIN-UNet was adapted from
https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/networks/swin_transformer_unet_skip_expand_decoder_sys.py
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

import healpix as hp

from lib.dataspec import DataSpec

from experiments.weather.models import hp_shifting
from experiments.weather.models.hp_windowing_isolatitude import (
    window_partition,
    window_reverse,
)
from experiments.weather.data import DataSpecHP
from lib.serialize_human import serialize_human

INJECT_SAVE = None


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): Number of pixels in the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        input_resolution=None,
        rel_pos_bias=None,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cos_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.rel_pos_bias = rel_pos_bias

        if self.use_cos_attn:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        if self.rel_pos_bias == "earth":
            # B * n_windows, window_size, C
            n_windows = (
                torch.tensor(input_resolution).prod()
                // torch.tensor(window_size).prod()
            )
            # indices = torch.arange(end=window_size)
            # coords = indices[:, None] - indices[None, :]
            window_size_d, window_size_hp = window_size
            self.earth_position_bias = nn.Parameter(
                torch.zeros(
                    (
                        n_windows,
                        1,
                        # self.num_heads,
                        window_size_d * window_size_hp,
                        window_size_d * window_size_hp,
                    )
                )
            )
            trunc_normal_(self.earth_position_bias, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, window_size, window_size) or None
        """
        B_, W, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, W, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if self.use_cos_attn:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale,
                max=torch.log(torch.tensor(1.0 / 0.01, device=self.logit_scale.device)),
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if self.rel_pos_bias == "earth":
            attn = attn + self.earth_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, W, W) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, W, W)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Number of input pixels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        input_resolution,
        base_pix,
        num_heads,
        window_size=4,
        shift_size=0,
        shift_strategy="nest_roll",
        rel_pos_bias=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_v2_norm_placement=False,
        use_cos_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_v2_norm_placement = use_v2_norm_placement
        # if self.input_resolution <= self.window_size:
        # if window size is larger than input resolution, we don't partition windows
        # self.shift_size = 0
        # self.window_size = self.input_resolution

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            input_resolution=input_resolution,
            window_size=window_size,
            num_heads=num_heads,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cos_attn=use_cos_attn,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0,
        )
        self.conv_kernel_size = 9
        self.conv = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.conv_kernel_size,
            padding=(self.conv_kernel_size - 1) // 2,
        )
        self.conv_bn = torch.nn.BatchNorm1d(dim)
        self.norm_conv = norm_layer(dim)

        # get nside parameter of current resolution
        nside = math.sqrt(input_resolution[1] // base_pix)
        # assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = math.floor(nside)
        # shifter classes and arguments for their init functions
        # separate this so only the needed class gets instantiated
        shifters = {
            "nest_roll": (
                hp_shifting.NestRollShift,
                {
                    "shift_size": self.shift_size,
                    "input_resolution": self.input_resolution,
                    "window_size": self.window_size,
                },
            ),
            "nest_grid_shift": (
                hp_shifting.NestGridShift,
                {"nside": nside, "base_pix": base_pix, "window_size": self.window_size},
            ),
            "ring_shift": (
                hp_shifting.RingShift,
                {
                    "nside": nside,
                    "base_pix": base_pix,
                    "window_size": self.window_size,
                    "shift_size": self.shift_size,
                    "input_resolution": self.input_resolution,
                },
            ),
        }

        if self.shift_size > 0:
            self.shifter = shifters[shift_strategy][0](**shifters[shift_strategy][1])
        else:
            self.shifter = hp_shifting.NoShift()

        attn_mask = self.shifter.get_mask(
            lambda x, window_size: window_partition(
                x, window_size, device=next(self.parameters()).device
            )
        )

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # N = self.input_resolution
        B, D, N, C = x.shape

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        B, D, N, C = x.shape
        # before_conv = x
        x = x.reshape(-1, N, C)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(B, D, N, C)

        # cyclic shift
        # breakpoint()
        # x[0, 0, 0:16, 0] = 15
        # x[0, 0, (16 * 100) : (16 * 101), 0] = 10
        # x[0, 0, -16:, 0] = 5
        # x[0, 0, :, 0] = torch.arange(0, N)

        # if self.shift_size > 0:
        #     INJECT_SAVE(
        #         "pre_shift_vis_windows_deep.npy", x[0, 0, ...].detach().permute(1, 0)
        #     )

        shifted_x = self.shifter.shift(x)
        # debug_shifted_x = shifted_x

        # if self.shift_size > 0:
        #     # global save_and_register
        #     INJECT_SAVE(
        #         "post_shift_vis_windows_deep.npy",
        #         shifted_x[0, -1, ...].detach().permute(1, 0),
        #     )

        # breakpoint()
        # shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size, device=next(self.parameters()).device
        )  # nW*B, window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(
            attn_windows, self.window_size, D, N, device=next(self.parameters()).device
        )  # B N' C

        # reverse cyclic shift
        x = self.shifter.shift_back(shifted_x)
        # x = shifted_x
        # if self.shift_size > 0:
        #     inverse_shift = self.shifter.shift_back(debug_shifted_x)
        #     INJECT_SAVE(
        #         "inverse_shifted_back_vis.npy",
        #         inverse_shift[0, 0, ...].detach().permute(1, 0),
        #     )

        # x = self.norm_conv(x)
        # x = before_conv + x
        # x = torch.nn.functional.gelu(x)

        # breakpoint()

        # FFN
        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads},"
            f" window_size={self.window_size}, shift_size={self.shift_size}"
            f", mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, N, C
        """
        B, D, N, C = x.shape
        assert (
            N % 4 == 0
        ), f"x size {N} is not divisible by 4 as necessary for patching."

        x0 = x[:, :, 0::4, :]  # B N/4 C
        x1 = x[:, :, 1::4, :]  # B N/4 C
        x2 = x[:, :, 2::4, :]  # B N/4 C
        x3 = x[:, :, 3::4, :]  # B N/4 C
        # concatenate the patches per merge-window channel-wise
        x = torch.cat([x0, x1, x2, x3], -1)  # B N/4 patch_size*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        dim: input channels
        dim_scale: upscaling factor for channels before patch expansion
        """
        super().__init__()
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale != 1 else nn.Identity()
        )
        self.norm = norm_layer(dim // 2)
        self.linear = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(self, x):
        """
        x: B, N, dim
        """
        x = self.expand(x)
        B, D, N, C = x.shape
        # breakpoint()
        # TODO Let's do shifted upsample and combine?
        x = rearrange(x, "b d n (p c)-> b d (n p) c", p=4, c=C // 4)
        x = self.norm(x)
        x = self.linear(x)

        return x


class FinalPatchExpand_Transpose(nn.Module):
    def __init__(self, patch_size, dim, data_spec_hp: DataSpecHP):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.conv_surface = nn.ConvTranspose1d(
            dim,
            data_spec_hp.n_surface,
            kernel_size=patch_size,
            stride=patch_size,
            # padding=patch_size + 2,
        )
        self.conv_upper = nn.ConvTranspose2d(
            dim,
            data_spec_hp.n_upper,
            kernel_size=[2, patch_size],
            stride=[2, patch_size],
            # padding=[0, patch_size + 2],
        )
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # B D N C -> B C D N
        x = x.permute(0, 3, 1, 2)
        # breakpoint()
        x_surface = self.conv_surface(x[:, :, 0, :])
        x_upper = self.conv_upper(x[:, :, 1:, :])
        x_surface = x_surface.permute(0, 2, 1)
        x_upper = x_upper.permute(0, 2, 3, 1)
        # x = self.norm(x)
        return x_surface, x_upper


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, patch_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, N, dim
        """
        x = self.expand(x)
        B, N, C = x.shape

        x = rearrange(
            x, "b n (p c)-> b (n p) c", p=self.patch_size, c=C // self.patch_size
        )
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Number of pixels in input.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        base_pix,
        shift_size,
        shift_strategy,
        rel_pos_bias,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        use_v2_norm_placement=False,
        use_cos_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    base_pix=base_pix,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    shift_strategy=shift_strategy,
                    rel_pos_bias=rel_pos_bias,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    use_v2_norm_placement=use_v2_norm_placement,
                    use_cos_attn=use_cos_attn,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, config, data_spec: DataSpecHP):
        super().__init__()
        # assert config.patch_size % 4 == 0, "required for valid nside in deeper layers"

        self.config = config
        self.data_spec = data_spec

        self.num_hp_patches = hp.nside2npix(data_spec.nside) // config.patch_size

        self.proj_surface = nn.Conv1d(
            data_spec.n_surface,
            config.embed_dims[0],
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.proj_upper = nn.Conv2d(
            data_spec.n_upper,
            config.embed_dims[0],
            kernel_size=[2, config.patch_size],
            stride=[2, config.patch_size],
        )

    def forward(self, x_surface, x_upper):
        B, C, N = x_surface.shape
        assert N == hp.nside2npix(
            self.data_spec.nside
        ), f"Input image size ({N}) doesn't match model ({self.data_spec.input_shape[0]})."
        x_surface = self.proj_surface(x_surface)[:, :, None, :]
        x_upper = self.proj_upper(x_upper)
        x_upper = torch.nn.functional.pad(x_upper, (0, 0, 1, 0))
        x = torch.concatenate([x_surface, x_upper], dim=2)
        # breakpoint()
        x = x.permute(0, 2, 3, 1)
        return x


@dataclass
class SwinHPPanguIsolatitudeConvConfig:
    base_pix: int = 12
    nside: int = 64
    patch_size: int = 16
    window_size: int = 36
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    patch_embed_norm_layer: Optional[Literal[nn.LayerNorm]] = None
    depths: List[int] = field(default_factory=lambda: [2, 6, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [6, 12, 12, 6])
    embed_dims: List[int] = field(default_factory=lambda: [192, 384, 384, 192])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal[nn.LayerNorm] = nn.LayerNorm
    use_v2_norm_placement: bool = False
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    dev_mode: bool = False  # Developer mode for printing extra information
    # decoder_class: Literal[UnetDecoder] = UnetDecoder

    def serialize_human(self):
        return serialize_human(self.__dict__)  # dict(validation=self.validation)


class SwinHPPanguIsolatitudeConv(nn.Module):
    r"""Swin Transformer A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer
        using Shifted Windows` - https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer

    """

    def __init__(
        self, config: SwinHPPanguIsolatitudeConvConfig, data_spec: DataSpec, **kwargs
    ):
        super().__init__()

        self.config = config
        self.data_spec = data_spec

        self.num_layers = len(config.depths)
        # self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        # self.num_features_up = int(config.embed_dim * 2)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(config, data_spec=data_spec)
        num_hp_patches = self.patch_embed.num_hp_patches
        self.input_resolutions = [
            [8, num_hp_patches],
            [8, num_hp_patches // 4],
            [8, num_hp_patches // 4],
            [8, num_hp_patches],
        ]

        # absolute position embedding
        # if config.ape:
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros(1, num_patches, config.embed_dims[0])
        #     )
        #     trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.downsample = PatchMerging(config.embed_dims[0], dim_scale=2)
        self.upsample = PatchExpand(config.embed_dims[1], dim_scale=2)
        # self.final_up = FinalPatchExpand_X4(
        #     patch_size=config.patch_size,
        #     dim=2 * config.embed_dims[-1],
        # )
        self.final_up = FinalPatchExpand_Transpose(
            patch_size=config.patch_size,
            dim=2 * config.embed_dims[-1],
            data_spec_hp=data_spec,
        )
        # self.final_up = FinalPatchExpand_Transpose(
        #     patch_size=config.patch_size, dim=config.embed_dims[0]
        # )
        # self.output = nn.Conv1d(
        #     in_channels=2 * config.embed_dims[-1],
        #     # in_channels=config.embed_dims[0],  # 2 * config.embed_dims[-1],
        #     out_channels=1,  # data_spec.output_shape[0],
        #     kernel_size=1,
        #     bias=False,
        # )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # if config.decoder_class == UnetDecoder:
            # downsample = PatchMerging if (i_layer < self.num_layers - 1) else None

            layer = BasicLayer(
                dim=config.embed_dims[i_layer],
                input_resolution=self.input_resolutions[i_layer],
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                base_pix=config.base_pix,
                shift_size=config.shift_size,
                shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[
                    sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                ],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)

        # out_channels = self.num_features
        self.norm = config.norm_layer(config.embed_dims[1])
        # self.mlp_in = torch.nn.Linear(
        #     data_spec.input_shape[1] // config.patch_size * config.embed_dims[0],
        #     # 147456,
        #     # data_spec.input_shape[0] // config.patch_size * config.embed_dims[0],
        #     256,
        #     bias=True,
        # )
        # self.mlps = torch.nn.ModuleList(
        #     [torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in [(256, 256)]]
        # )
        # self.mlp_out = torch.nn.Linear(
        #     256,
        #     data_spec.input_shape[1] // config.patch_size * config.embed_dims[0],
        #     bias=True,
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def _forward(self, batch_tuple):  # , batch):
        x_surface, x_upper = batch_tuple  # ["input_surface"]
        layer_out = []
        # x_upper = batch["input_upper"]
        # input = x
        x = self.patch_embed(x_surface, x_upper)
        # layer_out.append(("patch_embed", x[0, 0, :, 0].detach()))
        # skip2 = x
        # x = x + self.absolute_pos_embed
        # x = x.permute(0, 2, 1)
        x = self.layers[0](x)
        # layer_out.append(("layer0", x[0, 0].permute(1, 0).detach()))
        skip = x
        x = self.downsample(x)
        # layer_out.append(("downsample", x[0, 0, :, 0].detach()))
        x = self.layers[1](x)
        # layer_out.append(("layer1", x[0, 0].permute(1, 0).detach()))
        x = self.layers[2](x)
        # layer_out.append(("layer2", x[0, 0].permute(1, 0).detach()))
        x = self.norm(x)
        # # layer_out.append(("norm_c0", x[0, 0, :, 0].detach()))
        # # layer_out.append(("norm_c10", x[0, 0, :, 10].detach()))
        # # layer_out.append(("norm_c95", x[0, 0, :, 95].detach()))
        # # layer_out.append(("norm_c50", x[0, 0, :, 50].detach()))
        x = self.upsample(x)
        # layer_out.append(("upsample", x[0, 0].permute(1, 0).detach()))
        x = self.layers[3](x)
        # layer_out.append(("layer3", x[0, 0].permute(1, 0).detach()))
        x = torch.concatenate([skip, x], -1)
        # layer_out.append(("post_skip", x[0, 0].permute(1, 0).detach()))
        # breakpoint()
        # x = torch.concatenate([x, x], -1)
        x_surface, x_upper = self.final_up(x)
        # layer_out.append(("final_up_surface", x_surface[0, :, 0].detach()))
        # layer_out.append(("final_up_upper", x_upper[0, 0, :, 0].detach()))
        x_surface = x_surface.permute(0, 2, 1)
        x_upper = x_upper.permute(0, 3, 1, 2)
        return x_surface, x_upper[:, :, :13, :], layer_out

    def _forward_debug(self, batch_tuple):  # , batch):
        x_surface, x_upper = batch_tuple  # ["input_surface"]
        layer_out = []
        # x_upper = batch["input_upper"]
        # input = x
        x = self.patch_embed(x_surface, x_upper)
        layer_out.append(("patch_embed", x[0, 0, :, 0].detach()))
        # skip2 = x
        # x = x + self.absolute_pos_embed
        # x = x.permute(0, 2, 1)
        x = self.layers[0](x)
        layer_out.append(("layer0", x[0, 0].permute(1, 0).detach()))
        skip = x
        x = self.downsample(x)
        layer_out.append(("downsample", x[0, 0, :, 0].detach()))
        x = self.layers[1](x)
        layer_out.append(("layer1", x[0, 0].permute(1, 0).detach()))
        x = self.layers[2](x)
        layer_out.append(("layer2", x[0, 0].permute(1, 0).detach()))
        x = self.norm(x)
        # layer_out.append(("norm_c0", x[0, 0, :, 0].detach()))
        # layer_out.append(("norm_c10", x[0, 0, :, 10].detach()))
        # layer_out.append(("norm_c95", x[0, 0, :, 95].detach()))
        # layer_out.append(("norm_c50", x[0, 0, :, 50].detach()))
        x = self.upsample(x)
        layer_out.append(("upsample", x[0, 0].permute(1, 0).detach()))
        x = self.layers[3](x)
        layer_out.append(("layer3", x[0, 0].permute(1, 0).detach()))
        x = torch.concatenate([skip, x], -1)
        layer_out.append(("post_skip", x[0, 0].permute(1, 0).detach()))
        # breakpoint()
        # x = torch.concatenate([x, x], -1)
        x_surface, x_upper = self.final_up(x)
        layer_out.append(("final_up_surface", x_surface[0, :, 0].detach()))
        layer_out.append(("final_up_upper", x_upper[0, 0, :, 0].detach()))
        x_surface = x_surface.permute(0, 2, 1)
        x_upper = x_upper.permute(0, 3, 1, 2)
        return x_surface, x_upper[:, :, :13, :], layer_out

    def forward_tuple(self, batch):
        surface, upper, _ = self._forward(
            (batch["input_surface"], batch["input_upper"])
        )
        return surface, upper

    def forward(self, batch):
        # x = x.permute(0, 2, 1)  # B,C,N
        # x = self.output(x)
        x_surface, x_upper, layer_out = self._forward(
            (batch["input_surface"], batch["input_upper"])
        )

        return dict(logits_surface=x_surface, logits_upper=x_upper)

    def forward_debug(self, batch):
        # x = x.permute(0, 2, 1)  # B,C,N
        # x = self.output(x)
        x_surface, x_upper, layer_out = self._forward_debug(
            (batch["input_surface"], batch["input_upper"])
        )

        # return dict(logits_surface=x_surface, logits_upper=x_upper)
        return dict(logits_surface=x_surface, logits_upper=x_upper, layer_out=layer_out)
