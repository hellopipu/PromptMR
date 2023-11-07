'''
Author: Bingyu Xin
Affiliation: Computer Science department, Rutgers University, NJ, USA
Paper: https://arxiv.org/abs/2309.13839
Date: 2023-10-15
'''

import torch
import torch.nn as nn
import fastmri
import math
import torch.nn.functional as F
from fastmri.data import transforms
from typing import (
    List,
    Optional,
    Tuple,
)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_input_prompt = False):
        super(PromptBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(
            1, prompt_len, prompt_dim, prompt_size, prompt_size), requires_grad=learnable_input_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act, no_use_ca=False, first_act=False):
        super(DownBlock, self).__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction, bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(*(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(UpBlock, self).__init__()

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)


    def forward(self,x,prompt_dec,skip):

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super(SkipBlock, self).__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(
                *[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)

        return x


class PromptUnet(nn.Module):
    def __init__(self, 
                 in_chans=10, 
                 out_chans=10, 
                 n_feat0=48,
                 feature_dim = [72, 96, 120],
                 prompt_dim = [24, 48, 72],
                 len_prompt = [5, 5, 5],
                 prompt_size = [64, 32, 16],
                 n_enc_cab = [2, 3, 3],
                 n_dec_cab = [2, 2, 3],
                 n_skip_cab = [1, 1, 1],
                 n_bottleneck_cab = 3,
                 no_use_ca = False,
                 learnable_input_prompt=False,
                 kernel_size=3, 
                 reduction=4, 
                 act=nn.PReLU(), 
                 bias=False,
                 ):
        """
        PromptUnet, see in paper: https://arxiv.org/abs/2309.13839
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            n_feat0: Number of output channels in the first convolution layer.
            feature_dim: Number of output channels in each level of the encoder.
            prompt_dim: Number of channels in the prompt at each level of the decoder.
            len_prompt: number of components in the prompt at each level of the decoder.
            prompt_size: spatial size of the prompt at each level of the decoder.
            n_enc_cab: number of channel attention blocks (CAB) in each level of the encoder.
            n_dec_cab: number of channel attention blocks (CAB) in each level of the decoder.
            n_skip_cab: number of channel attention blocks (CAB) in each skip connection.
            n_bottleneck_cab: number of channel attention blocks (CAB) in the bottleneck.
            kernel_size: kernel size of the convolution layers.
            reduction: reduction factor for the channel attention blocks (CAB).
            act: activation function.
            bias: whether to use bias in the convolution layers.
            no_use_ca: whether to *not* use channel attention blocks (CAB).
            learnable_input_prompt: whether to learn the input prompt in the PromptBlock.
        """
        super(PromptUnet, self).__init__()

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_bottleneck_cab)])

        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level2 = PromptBlock(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        self.prompt_level1 = PromptBlock(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca=no_use_ca)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x):
        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x,dec_prompt3,self.skip_attn3(enc3))

        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x,dec_prompt2,self.skip_attn2(enc2))

        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x,dec_prompt1,self.skip_attn1(enc1))

        # 4. last conv
        return self.conv_last(x)


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int = 10,
        out_chans: int = 10,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_input_prompt=False,
    ):

        super().__init__()
        self.unet = PromptUnet(in_chans=in_chans,
                                out_chans = out_chans, 
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca,
                                learnable_input_prompt=learnable_input_prompt,
                                )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class PromptMRBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, num_adj_slices=5):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        _, c, _, _, _ = sens_maps.shape
        return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // self.num_adj_slices, dim=1), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, c, h, w, _ = x.shape
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(b, self.num_adj_slices, c // self.num_adj_slices, h, w, 2).sum(
            dim=2, keepdim=False
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace -
                              ref_kspace, zero) * self.dc_weight

        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        num_adj_slices: int = 5,
        n_feat0: int = 24,
        feature_dim: List[int] = [36, 48, 60],
        prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        mask_center: bool = True,

    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.norm_unet = NormPromptUnet(in_chans=in_chans,
                                out_chans = out_chans,
                                n_feat0=n_feat0,
                                feature_dim = feature_dim,
                                prompt_dim = prompt_dim,
                                len_prompt = len_prompt,
                                prompt_size = prompt_size,
                                n_enc_cab = n_enc_cab,
                                n_dec_cab = n_dec_cab,
                                n_skip_cab = n_skip_cab,
                                n_bottleneck_cab = n_bottleneck_cab,
                                no_use_ca = no_use_ca)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / fastmri.rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
        # convert to image space
        images, batches = self.chans_to_batch_dim(
            fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class PromptMR(nn.Module):
    """
    An prompt-learning based unrolled model for multi-coil MR reconstruction, 
    see https://arxiv.org/abs/2309.13839.

    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],
        prompt_dim: List[int] = [24, 48, 72],
        sens_n_feat0: int =24,
        sens_feature_dim: List[int] = [36, 48, 60],
        sens_prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [5, 5, 5],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in
                BottleneckBlock.
            no_use_ca: not using channel attention.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
        """
        super().__init__()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim= sens_feature_dim,
            prompt_dim = sens_prompt_dim,
            len_prompt = sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size = sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab = sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab = sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab = sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab = sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca = sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,

        )
        self.cascades = nn.ModuleList(
            [PromptMRBlock(NormPromptUnet(2*num_adj_slices, 2*num_adj_slices, n_feat0, feature_dim, prompt_dim, len_prompt, prompt_size, n_enc_cab, n_dec_cab, n_skip_cab, n_bottleneck_cab, no_use_ca), num_adj_slices) for _ in range(num_cascades)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, num_low_frequencies, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()
        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                kspace_pred = torch.utils.checkpoint.checkpoint(
                cascade, kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False)
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[
            self.center_slice]

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)