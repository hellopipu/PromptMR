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

    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0=48,
        feature_dim = [72, 96, 120],
        prompt_dim = [24, 48, 72],
        sens_n_feat0=24,
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
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        print(locals())
        self.save_hyperparameters()
        assert num_adj_slices % 2 == 1, "num_adj_slices must be odd"

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.no_use_ca = no_use_ca

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
            use_checkpoint=use_checkpoint,
            no_use_ca=self.no_use_ca,
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

        return parser
