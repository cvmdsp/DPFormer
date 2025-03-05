import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange


def drop_path(x, drop_prob: float = 0.1, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=30):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


# MLP块
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_chans, img_size=64, patch_size=1, embed_dim=48, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=48):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class BottleneckPool(nn.Module):
    def __init__(self, dim):
        super(BottleneckPool, self).__init__()
        self.dim = dim

        self.max_pool = nn.MaxPool2d(2)
        self.act = nn.LeakyReLU()

    def forward(self, x, exp):
        for _ in range(exp):
            x = self.max_pool(x)
        x = self.act(x)
        return x


class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SpaTB(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.norm1 = norm_layer(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.drop_path = DropPath()

    def forward(self, x, y, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        y = self.norm1(y)
        y = y.view(b, h, w, c)
        kv = self.kv(x).reshape(b, h, w, 2, c).permute(3, 0, 4, 1, 2)
        q = self.q(y)
        kv = torch.cat((kv[0], kv[1]), dim=1)

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)
        kv_windows = self.unfold(kv)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # 256,6,16,16
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,6,36,16
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d

        # windows attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SpeTB(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 compress_ratio,
                 squeeze_factor,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size  # (4*0.5)+4=6  overlap的窗口尺寸

        self.norm1 = norm_layer(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # nn.Unfold 用于在输入的张量上执行滑动窗口操作
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath()

    def forward(self, x, y, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = y
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        y = self.norm1(y)
        y = y.view(b, h, w, c)
        q = self.q(x)  # b, h, w, c
        k = self.k(y)
        v = self.v(y)
        qk = torch.cat((q, k), dim=-1)  # b, h, w, 2*c
        qk = qk.permute(0, 3, 1, 2)

        # partition windows
        v_windows = window_partition(v, self.window_size)  # nw*b, window_size, window_size, c
        v_windows = v_windows.view(-1, self.window_size * self.window_size, c)  # （256, 16, 48）
        qk_windows = self.unfold(qk)  # b, c*w*w, nw
        qk_windows = rearrange(qk_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        q_windows, k_windows = qk_windows[0], qk_windows[1]
        b_, nq, _ = v_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,36,12
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,36,12
        v = v_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # 256,4,16,12

        # windows attention
        q = q * self.scale
        attn = (q.transpose(-2, -1) @ k)
        attn = self.softmax(attn)  # (256, 4, 12, 12)
        attn_windows = (v @ attn).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)  # 256,4,4,96
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut
        x_ = self.norm2(x)
        x_ = x_.view(b, h, w, c)
        conv_x = self.drop_path(self.conv_block(x_.permute(0, 3, 1, 2)))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        x = conv_x + x

        return x


class BLOCK(nn.Module):

    def __init__(self,
                 compress_ratio,
                 squeeze_factor,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size,
                 overlap_ratio,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.SpaTB = SpaTB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        self.SpeTB = SpeTB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

    def forward(self, x, y, x_size, params):
        x = self.SpaTB(x, y, x_size, params['rpi'])
        y = self.SpeTB(x, y, x_size)

        return x, y


class LAYER(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 overlap_ratio):
        super().__init__()
        self.dim = dim,
        self.input_resolution = input_resolution,
        self.depth = depth,
        self.num_heads = num_heads,
        self.window_size = window_size,
        self.compress_ratio = compress_ratio,
        self.squeeze_factor = squeeze_factor,
        self.overlap_ratio = overlap_ratio,

        self.blocks = nn.ModuleList([
            BLOCK(
                compress_ratio,
                squeeze_factor,
                dim,
                input_resolution,
                num_heads,
                window_size,
                overlap_ratio,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=4,
                norm_layer=nn.LayerNorm) for i in range(depth)
        ])

    def forward(self, x, y, x_size, params):
        for blk in self.blocks:
            x, y = blk(x, y, x_size, params)

        return x, y


class DPTRB(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 overlap_ratio,
                 img_size=64,
                 patch_size=1,
                 resi_connection='1conv'):
        super(DPTRB, self).__init__()

        self.layer = LAYER(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim)

    def forward(self, x, y, x_size, params):

        x, y = self.layer(x, y, x_size, params)
        x = self.patch_embed(self.conv(self.patch_unembed(x, x_size))) + x
        y = self.patch_embed(self.conv(self.patch_unembed(y, x_size))) + y

        return x, y


@ARCH_REGISTRY.register()
class DPFormer(nn.Module):

    def __init__(self,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 embed_dim=48,
                 img_size=64,
                 patch_size=1,
                 depths=1,
                 depths0=2,
                 num_heads=4,
                 window_size=4,
                 compress_ratio=3,
                 squeeze_factor=30,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv'):

        super(DPFormer, self).__init__()

        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_range = img_range

        self.weight = nn.Parameter(torch.tensor([0.5]))
        self.norm = nn.LayerNorm(embed_dim)
        self.Conv5_dim = nn.Conv2d(in_channels=n_select_bands, out_channels=embed_dim, kernel_size=3, padding=(1, 1))
        self.Convin_dim = nn.Conv2d(in_channels=n_bands, out_channels=embed_dim, kernel_size=3, padding=(1, 1))
        self.pool = BottleneckPool(embed_dim)
        self.upsample1 = Upsample(upscale, embed_dim)
        self.upsample2 = Upsample(upscale * 2, embed_dim)
        self.depthwise = nn.Conv2d(embed_dim * 3, embed_dim * 3, 3, 1, 1, groups=embed_dim * 3)
        self.act = nn.GELU()
        self.pointwise = nn.Linear(embed_dim * 3, embed_dim)
        self.mean = torch.zeros(1, 1, 1, 1)

        num_feat = 64  # Only used once in the final reconstruction module

        relative_position_index = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index', relative_position_index)

        # 图像大小缩小四倍
        self.D0 = nn.Sequential(
            nn.Conv2d(embed_dim, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 图像大小缩小四倍
        self.D1 = nn.Sequential(
            nn.Conv2d(embed_dim, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # split image into non-overlapping patches
        self.patch_embed0 = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        self.patch_embed1 = PatchEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        patches_resolution0 = self.patch_embed0.patches_resolution
        self.patches_resolution0 = patches_resolution0

        patches_resolution1 = self.patch_embed1.patches_resolution
        self.patches_resolution1 = patches_resolution1

        patches_resolution2 = self.patch_embed2.patches_resolution
        self.patches_resolution2 = patches_resolution2

        # merge non-overlapping patches into image
        self.patch_unembed0 = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)

        self.patch_unembed1 = PatchUnEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)

        self.patch_unembed2 = PatchUnEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim)

        self.DPTRB0 = DPTRB(
            dim=embed_dim,
            input_resolution=(patches_resolution0[0], patches_resolution0[1]),
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )
        self.DPTRB1 = DPTRB(
            dim=embed_dim,
            input_resolution=(patches_resolution1[0], patches_resolution1[1]),
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size // 2,
            patch_size=patch_size,
            resi_connection=resi_connection
        )
        self.DPTRB2 = DPTRB(
            dim=embed_dim,
            input_resolution=(patches_resolution2[0], patches_resolution2[1]),
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size // 4,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.DPTRB3 = DPTRB(
            dim=embed_dim,
            input_resolution=(patches_resolution0[0], patches_resolution0[1]),
            depth=depths0,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            overlap_ratio=overlap_ratio,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        # Reconstruction module: convolution + activation + convolution
        self.conv_before = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))  # num_feat=64 (通道数从48到64)

        self.conv_last = nn.Conv2d(num_feat, n_bands, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_oca(self):

        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ori_flatten = torch.flatten(coords_ori, 1)

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_ext_flatten = torch.flatten(coords_ext, 1)

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features0(self, x, y):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index}

        x = self.patch_embed0(x)
        y = self.patch_embed0(y)

        x, y = self.DPTRB0(x, y, x_size, params)

        x = self.norm(x)
        y = self.norm(y)
        x = self.patch_unembed0(x, x_size)
        y = self.patch_unembed0(y, x_size)

        return x, y

    def forward_features1(self, x, y):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index}

        x = self.patch_embed1(x)
        y = self.patch_embed1(y)

        x, y = self.DPTRB1(x, y, x_size, params)

        x = self.norm(x)
        y = self.norm(y)
        x = self.patch_unembed1(x, x_size)
        y = self.patch_unembed1(y, x_size)

        return x, y

    def forward_features2(self, x, y):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index}

        x = self.patch_embed2(x)
        y = self.patch_embed2(y)

        x, y = self.DPTRB2(x, y, x_size, params)

        x = self.norm(x)
        y = self.norm(y)
        x = self.patch_unembed2(x, x_size)
        y = self.patch_unembed2(y, x_size)

        return x, y

    def forward_features3(self, x, y):
        x_size = (x.shape[2], x.shape[3])

        params = {'rpi': self.relative_position_index}

        x = self.patch_embed0(x)
        y = self.patch_embed0(y)

        x, y = self.DPTRB3(x, y, x_size, params)

        x = self.norm(x)
        y = self.norm(y)
        x = self.patch_unembed0(x, x_size)
        y = self.patch_unembed0(y, x_size)

        return x, y

    def forward(self, LR_HSI, HR_MSI):
        lms = F.interpolate(LR_HSI, scale_factor=self.scale_ratio, mode='bilinear')

        x = self.Conv5_dim(HR_MSI)
        y = self.Convin_dim(lms)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        self.mean = self.mean.type_as(y)
        y = (y - self.mean) * self.img_range

        x0, y0 = self.forward_features0(x, y)
        cat0 = torch.add(x0, y0)
        x0 = self.D0(x0)
        y0 = self.D0(y0)

        x1, y1 = self.forward_features1(x0, y0)
        cat1 = torch.add(x1, y1)
        x1 = self.D1(x1)
        y1 = self.D1(y1)

        x2, y2 = self.forward_features2(x1, y1)
        cat2 = torch.add(x2, y2)

        z1 = torch.add(cat0, y)
        z2 = self.upsample1(torch.add(cat1, self.pool(y, 1)))
        z3 = self.upsample2(torch.add(cat2, self.pool(y, 2)))

        z = torch.cat([z1, z2, z3], dim=1)

        z = self.act(self.depthwise(z))
        height = z.size(-2)
        width = z.size(-1)
        z = rearrange(z, 'b c h w -> b (h w) c')
        z = self.pointwise(z)
        z = self.norm(z)
        z = rearrange(z, 'b (h w) c -> b c h w', h=height, w=width)
        z_0, z_1 = self.forward_features3(z, z)
        fuse0 = z_0 + z_1

        x = self.conv_before(fuse0)
        x = self.conv_last(x)
        x = x + lms

        x = x / self.img_range + self.mean

        return x
