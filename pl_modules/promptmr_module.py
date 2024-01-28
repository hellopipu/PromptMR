'''
Author: Bingyu Xin
Affiliation: Computer Science department, Rutgers University, NJ, USA
Paper: https://arxiv.org/abs/2309.13839
Date: 2023-10-15
'''

from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from fastmri.pl_modules import MriModule
from models.promptmr import PromptMR
from typing import List

class PromptMrModule(MriModule):
    """
    PromptMR training module.

    Paper: https://arxiv.org/abs/2309.13839

    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int =48,
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
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        use_checkpoint: bool = False,
        low_mem: bool = False,
        **kwargs,
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
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            low_mem: Whether to compute sensitivity map coil by coil to save GPU memory.

        """
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca
        self.use_checkpoint = use_checkpoint
        self.low_mem = low_mem

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            n_feat0=self.n_feat0,
            feature_dim = self.feature_dim,
            prompt_dim = self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim = self.sens_feature_dim,
            sens_prompt_dim = self.sens_prompt_dim,
            len_prompt = self.len_prompt,
            prompt_size = self.prompt_size,
            n_enc_cab = self.n_enc_cab,
            n_dec_cab = self.n_dec_cab,
            n_skip_cab = self.n_skip_cab,
            n_bottleneck_cab = self.n_bottleneck_cab,
            no_use_ca = self.no_use_ca,
            use_checkpoint=self.use_checkpoint,
            low_mem = self.low_mem
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.promptmr(masked_kspace, mask, num_low_frequencies)

    def training_step(self, batch, batch_idx):

        output = self(batch.masked_kspace, batch.mask,
                      batch.num_low_frequencies)

        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies
        )
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self(batch.masked_kspace, batch.mask,
                      batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of PromptMR cascades",
        )
        parser.add_argument(
            "--num_adj_slices",
            default=5,
            type=int,
            help="Number of adjacent slices",
        )
        parser.add_argument(
            "--n_feat0",
            default=48,
            type=int,
            help="Number of PromptUnet top-level feature channels in PromptMR blocks",
        )
        parser.add_argument(
            "--feature_dim",
            default=[72, 96, 120],
            nargs="+",
            type=int,
            help="feature dim for each level in PromptUnet",
        )
        parser.add_argument(
            "--prompt_dim",
            default=[24, 48, 72],
            nargs="+",
            type=int,
            help="prompt dim for each level in PromptUnet in sensitivity map estimation (SME) network",
        )
        parser.add_argument(
            "--sens_n_feat0",
            default=24,
            type=int,
            help="Number of top-level feature channels for sense map estimation PromptUnet in PromptMR",
        )
        parser.add_argument(
            "--sens_feature_dim",
            default=[36, 48, 60],
            nargs="+",
            type=int,
            help="feature dim for each level in PromptUnet for sensitivity map estimation (SME) network",
        )
        parser.add_argument(
            "--sens_prompt_dim",
            default=[12, 24, 36],
            nargs="+",
            type=int,
            help="prompt dim for each level in PromptUnet in sensitivity map estimation (SME) network",
        )
        parser.add_argument(
            "--len_prompt",
            default=[5,5,5],
            nargs="+",
            type=int,
            help="number of prompt component in each level",
        )
        parser.add_argument(
            "--prompt_size",
            default=[64, 32, 16],
            nargs="+",
            type=int,
            help="prompt spatial size",
        )
        parser.add_argument(
            "--n_enc_cab",
            default=[2, 3, 3],
            nargs="+",
            type=int,
            help="number of CABs (channel attention Blocks) in DownBlock",
        )
        parser.add_argument(
            "--n_dec_cab",
            default=[2, 2, 3],
            nargs="+",
            type=int,
            help="number of CABs (channel attention Blocks) in UpBlock",
        )
        parser.add_argument(
            "--n_skip_cab",
            default=[1, 1, 1],
            nargs="+",
            type=int,
            help="number of CABs (channel attention Blocks) in SkipBlock",
        )
        parser.add_argument(
            "--n_bottleneck_cab",
            default=3,
            type=int,
            help="number of CABs (channel attention Blocks) in BottleneckBlock",
        )

        parser.add_argument(
            "--no_use_ca",
            default=False,
            action='store_true',
            help="not using channel attention",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--use_checkpoint", action="store_true", help="Use checkpoint (default: False)"
        )
        parser.add_argument(
            "--low_mem", action="store_true", help="consume less memory by computing sens_map coil by coil (default: False)"
        )

        return parser
