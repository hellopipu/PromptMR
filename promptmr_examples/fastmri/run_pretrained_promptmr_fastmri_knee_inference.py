
import argparse
import time
from collections import defaultdict

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import numpy as np
import requests
import torch
from tqdm import tqdm
from pathlib import Path
import fastmri
import fastmri.data.transforms as T

from models.promptmr import PromptMR
from pl_modules.data_module import FastmriKneeSliceDataset
from data.transforms import FastmriKneePromptMrDataTransform



VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def run_varnet_model(batch, model, device):
    crop_size = batch.crop_size

    output = model(batch.masked_kspace.to(device), batch.mask.to(device)).cpu()

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]

    return output, int(batch.slice_num[0]), batch.fname[0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )


def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )

def run_inference(challenge, state_dict_file, data_path, output_path, device):
    # model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    model = PromptMR(            
            num_cascades=12,  # number of unrolled iterations
            num_adj_slices=3,  # number of adjacent slices

            n_feat0=48,  # number of top-level channels for PromptUnet
            feature_dim = [72, 96, 120],
            prompt_dim = [24, 48, 72],

            sens_n_feat0=24,
            sens_feature_dim = [36, 48, 60],
            sens_prompt_dim = [12, 24, 36],
            
            # len_prompt = [5, 5, 5],
            # prompt_size = [64, 32, 16],
            # n_enc_cab = [2, 3, 3],
            # n_dec_cab = [2, 2, 3],
            # n_skip_cab = [1, 1, 1],
            # n_bottleneck_cab = 3,
            # lr=0.0001,  # AdamW learning rate;
            # lr_step_size=35,  # epoch at which to decrease learning rate
            # lr_gamma=0.1,  # extent to which to decrease learning rate
            # weight_decay=1e-2,  # weight regularization strength
            use_checkpoint=False,  # use checkpointing for GPU memory savings
            no_use_ca = True,
    )

    print('param: ', count_parameters(model))
    print('trainable param: ', count_trainable_parameters(model))
    print('untrainable param: ', count_untrainable_parameters(model))

    # download the state_dict if we don't have it
    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]
        model.load_state_dict(torch.load(state_dict_file))
    else:
    # model.load_state_dict(torch.load(state_dict_file))
        state_dict = torch.load(state_dict_file)['state_dict']
        state_dict.pop('loss.w')
        state_dict = {k.replace('promptmr.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.eval()

    # data loader setup
    data_transform = FastmriKneePromptMrDataTransform()
    dataset = FastmriKneeSliceDataset(
        root=data_path, transform=data_transform, challenge="multicoil"
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_varnet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="varnet_knee_mc",
        choices=(
            "varnet_knee_mc",
            "varnet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path for saving reconstructions",
    )

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )
