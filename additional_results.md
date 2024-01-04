# Evaluation Results on Calgary-Campinas Dataset and fastMRI Multi-coil Brain Dataset

Code and pretrained models will be released in the future. For the compared models below, their official pretrained models are used for evaluation.

## Calgary-Campinas Dataset

| Model              | Trained on | Training Acceleration | # of Params | PSNR/SSIM 5x | PSNR/SSIM 10x | Link                                                                |
| ------------------ | ---------- | --------------------- | ----------- | ------------ | ------------- | ------------------------------------------------------------------- |
| `Recurrent-VarNet` | train      | 5x and 10x            | 11M         | 36.27/0.9437 | -             | [official repo](https://github.com/NKI-AI/direct) checkpoint 148500 |
| `Recurrent-VarNet` | train      | 5x and 10x            | 11M         | -            | 33.27/0.9147  | [official repo](https://github.com/NKI-AI/direct) checkpoint 107000 |
| `PromptMR`         | train      | 5x and 10x            | 82M         | 36.83/0.9485 | 34.10/0.9269  | to be released                                                      |

We follow the data split used in `Recurrent-VarNet`. Evaluation is conducted using the code in the [official challenge repository](https://github.com/rmsouza01/MC-MRI-Rec), where the first and last 50 slices in each volume are excluded from the evaluation.

## FastMRI MultiCoil Brain Dataset

| Model        | Trained on  | Training Acceleration | # of Params | NMSE/PSNR/SSIM 4x   | NMSE/PSNR/SSIM 8x   | Link                                                                                           |
| ------------ | ----------- | --------------------- | ----------- | ------------------- | ------------------- | ---------------------------------------------------------------------------------------------- |
| `E2E-VarNet` | train + val | 4x and 8x             | 30M         | 0.0037/41.08/0.9591 | 0.0075/37.96/0.9423 | [official repo](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/varnet) |
| `PromptMR`   | train + val | 4x and 8x             | 78M         | 0.0033/41.59/0.9609 | 0.0063/38.82/0.9465 | to be released                                                                                 |

Note: the result of `E2E-VarNet` differs slightly from the one presented on the [official leaderboard](https://web.archive.org/web/20230324102125mp_/https://fastmri.org/leaderboards).