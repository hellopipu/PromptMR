
import os
import sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

from data.transforms import PromptMrDataTransform
from pl_modules.cmrxrecon_data_module import CmrxReconDataModule
from pl_modules.promptmr_module import PromptMrModule
from data.subsample import create_mask_for_mask_type
import pytorch_lightning as pl



def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, None, args.accelerations, args.center_numbers
    )
    # use equispaced_fixed masks for train transform, fixed masks for val transform
    train_transform = PromptMrDataTransform(mask_func=mask, use_seed=False)
    val_transform = PromptMrDataTransform(mask_func=mask)
    test_transform = PromptMrDataTransform()
    # ptl data module - this handles data loaders
    data_module = CmrxReconDataModule(
        data_path=args.data_path,
        h5py_folder=args.h5py_folder,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val, # combine train and val data for train
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.strategy in (
            "ddp_find_unused_parameters_false", "ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = PromptMrModule(
        num_cascades=args.num_cascades,
        num_adj_slices=args.num_adj_slices,
        n_feat0=args.n_feat0,
        feature_dim = args.feature_dim,
        prompt_dim = args.prompt_dim,

        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim = args.sens_feature_dim,
        sens_prompt_dim = args.sens_prompt_dim,

        len_prompt = args.len_prompt,
        prompt_size = args.prompt_size,
        n_enc_cab = args.n_enc_cab,
        n_dec_cab = args.n_dec_cab,
        n_skip_cab = args.n_skip_cab,
        n_bottleneck_cab = args.n_bottleneck_cab,

        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        use_checkpoint=args.use_checkpoint,
        no_use_ca = args.no_use_ca,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    # num_gpus = 2
    backend = "ddp_find_unused_parameters_false"
    batch_size = 1

    # set defaults based on optional directory config
    data_path = pathlib.Path('.')
    default_root_dir = data_path / "experiments"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    parser.add_argument(
        "--num_gpus",
        default=2,
        type=int,
        help="Number of GPUs to use",
    )

    parser.add_argument(
        "--exp_name",
        default="promptmr_train",
        type=str,
        help="experiment name",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction", 'equispaced_fixed'),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_numbers",
        nargs="+",
        default=[24],
        type=int,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config with path to fastMRI data and batch size
    parser = CmrxReconDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = PromptMrModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=12,  # number of unrolled iterations
        num_adj_slices=5,  # number of adjacent slices

        n_feat0=48,  # number of top-level channels for PromptUnet
        feature_dim = [72, 96, 120],
        prompt_dim = [24, 48, 72],

        sens_n_feat0=24,
        sens_feature_dim = [36, 48, 60],
        sens_prompt_dim = [12, 24, 36],

        len_prompt = [5, 5, 5],
        prompt_size = [64, 32, 16],
        n_enc_cab = [2, 3, 3],
        n_dec_cab = [2, 2, 3],
        n_skip_cab = [1, 1, 1],
        n_bottleneck_cab = 3,
        no_use_ca = False,
        lr=0.0002,  # AdamW learning rate;
        lr_step_size=11,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=1e-2,  # weight regularization strength
        use_checkpoint=False,  # use checkpointing for GPU memory savings
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        # gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
        # default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=12,  # max number of epochs
        gradient_clip_val=0.01
    )

    args = parser.parse_args()
    args.gpus = args.num_gpus # override pl.Trainer gpus arg
    acc_folder = "acc_" + "_".join(map(str, args.accelerations))
    args.default_root_dir = default_root_dir / args.exp_name / acc_folder
    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":

    run_cli()
