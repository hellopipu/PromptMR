# Reproducing PromptMR results on fastMRI MultiCoil Knee dataset

All the following commands should be run under the `promptmr_examples/fastmri` folder.

## Download the fastMRI MultiCoil Knee dataset

Download knee `multicoil_train` and `multicoil_val` set from fastMRI website: <https://fastmri.med.nyu.edu/>.

## Split the `multicoil_val` dataset

Since the online evaluation platform for original `multicoil_test` set is no longer available ([see here](
https://github.com/facebookresearch/fastMRI/discussions/293)), we can run the following command to automatically split original 199 cases in `multicoil_val` set into 99 validation cases and 100 test cases. Before running, please first rename the original `multicoil_val` folder to `multicoil_val_origin`. (If you have already downloaded the `multicoil_test`, rename that folder to `multicoil_test_origin` as well.) Then, download the sampling mask files I used for 100 splited test cases from [here](https://drive.google.com/file/d/1YY6fifXo5SNFLAhO5M6V9abmVgXs7NvJ/view?usp=sharing) (fastMRI `random` type sampling mask, 46 files for acc=4, 54 files for acc=8) . Finally, executing the following command will automatically generate our split `multicoil_val` and `multicoil_test` folders. Please modify the `data_path` and `mask_path` to reflect your own dataset and sampling mask file locations.

```bash
python split_val_test.py \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_val_origin \
--mask_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_mask
```

## Pretrained models

We provide Google Drive links for downloading our models trained on the fastMRI MultiCoil Knee train set.

| Model        |Trained on |Training Acceleration|# of Params     | NMSE/PSNR/SSIM 4x                | NMSE/PSNR/SSIM 8x              | Link                            |
|--------------|------------ |---------------------|----------------|----------------------------------|--------------------------------|--------------------------------------------------------------------------------------------|
| `E2E-VarNet` |train+val    |4x and 8x            |30M             |0.0053/39.37/0.9236               |0.0087/37.30/0.8936             |[official repo 1](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/varnet)                         |
| `HUMUS-Net`  |train        |8x only              |109M            |-                                 |0.0090/37.20/0.8946             |[official repo 2](https://github.com/z-fabian/HUMUS-Net)                                    |
| `HUMUS-Net-L`|train        |8x only              |228M            |-                                 |0.0086/37.45/0.8955             |[official repo 2](https://github.com/z-fabian/HUMUS-Net)                                    |
| `PromptMR`   |train        |8x only              |80M             |-                                 |0.0080/37.78/0.8983             |[my link1](https://drive.google.com/file/d/1HBlwrmOaMQycohznYrg-uu4GmiVfWjkL/view?usp=sharing) |
| `PromptMR`   |train        |4x and 8x            |80M             |**0.0051**/**39.71**/**0.9264**   |**0.0080**/**37.78**/**0.8984** |[my link2](https://drive.google.com/file/d/1afLCO3C_S4e-q7QCt04Ksmv34jETrFQ7/view?usp=sharing) |

Note: In this table, `train` and `val` means the original train set and validation set, respectively. Evaluations were performed on the split test set (100 cases), which is from original validation set (199 cases). `E2E-VarNet` and `HUMUS-Net(-L)` were assessed using their respective official pretrained models. However, `E2E-VarNet` has only made available the model that was trained on the combined training and original validation sets with acceleration factors of `4x` and `8x`. *This implies that our split test set was already exposed to `E2E-VarNet` during its training phase*. Consequently, the results of `E2E-VarNet` should be considered for reference purposes only and not subjected to rigorous comparison with other models.

You can directly use the following command to download our `PromptMR` models to the `pretrained_models` folder:

```bash
mkdir pretrained_models
cd pretrained_models
# model trained with acc=8 only
gdown 1HBlwrmOaMQycohznYrg-uu4GmiVfWjkL
# model trained with acc=4 and 8
gdown 1afLCO3C_S4e-q7QCt04Ksmv34jETrFQ7
```

## Inference

The following command will reproduce the results of the pretrained PromptMR model on the fastMRI MultiCoil Knee Test Set. Please modify `data_path` and `output_path` to reflect the locations of your dataset and output reconstruction, respectively. The `state_dict_file` specifies the path of the downloaded pretrained model.

```bash
CUDA_VISIBLE_DEVICES=0 python run_pretrained_promptmr_fastmri_knee_inference.py --challenge varnet_knee_mc \
--state_dict_file pretrained_models/promptmr-12cascades-epoch=43-step=764324.ckpt \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--output_path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee
```

## Evaluate

The following commands will evaluate the results of the pretrained PromptMR models on the fastMRI MultiCoil Knee `x8` Test Set. Please modify the `target-path`, `predictions-path` and `test-path` to your own path correspondingly. `acceleration` filters the test cases with the specific acceleration factor.

```bash
python evaluate.py \
--target-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_full \
--predictions-path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee/reconstructions \
--test-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--challenge multicoil \
--acceleration 8 
```

## Training

The following command will train the PromptMR model on the fastMRI MultiCoil Knee Training Set. Please modify the `data_path` to reflect your own dataset location. `center_fractions` specifies the low frequency fractions, while `accelerations` defines sampling rate of the mask. The `mask_type` is the type of the sampling mask (for the fastMRI knee dataset, we use the `random` type). Use `no_use_ca` to disable channel attention . The checkpoints and log files will be saved in the folder specified by `exp_name` . The `use_checkpoint` enables a [technique](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) that trades compute for memory, which is useful when GPU memory is limited.

```bash
# train with acc=8 only
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_fastmri.py \
--challenge multicoil \
--center_fractions 0.04 \
--accelerations 8 \
--mask_type random \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil \
--exp_name promptmr_train \
--num_gpus 2 \
--no_use_ca \
--use_checkpoint

# train with acc=4 and 8
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_fastmri.py \
--challenge multicoil \
--center_fractions 0.08 0.04 \
--accelerations 4 8 \
--mask_type random \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil \
--exp_name promptmr_train \
--num_gpus 2 \
--no_use_ca \
--use_checkpoint
```
