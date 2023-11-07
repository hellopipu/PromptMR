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

We provide Google Drive links for downloading our models trained on the fastMRI MultiCoil Knee Training Set with `x8` acceleration.

| Model              |# of Params     |Download Link                                                                              |
|--------------------|----------------|-------------------------------------------------------------------------------------------|
| PromptMR-12cascades|80M             |[Link](https://drive.google.com/file/d/1HBlwrmOaMQycohznYrg-uu4GmiVfWjkL/view?usp=sharing) |

You can also directly use the following command to download the models to the `pretrained_models` folder:

```bash
mkdir pretrained_models
cd pretrained_models
gdown 1HBlwrmOaMQycohznYrg-uu4GmiVfWjkL
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

The following command will train the PromptMR model on the fastMRI MultiCoil Knee Training Set with `x8` acceleration. Please modify the `data_path` to reflect your own dataset location. `center_fractions` specifies the low frequency fractions, while `accelerations` defines sampling rate of the mask. The `mask_type` is the type of the sampling mask (for the fastMRI knee dataset, we use the `random` type). Use `no_use_ca` to disable channel attention . The checkpoints and log files will be saved in the folder specified by `exp_name` . The `use_checkpoint` enables a [technique](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) that trades compute for memory, which is useful when GPU memory is limited.

```bash
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
```

## Results

<details>
<summary><strong>Quantitative Results</strong> (click to expand) </summary>

![fastMRI](../../assets/fastmri_quantitative.png)

Note: In this table, E2E-VarNet and HUMUS-Net(-L) were assessed using their respective official pretrained models. However, E2E-VarNet has only made available the model that was trained on the combined training and original validation sets with acceleration factors of `x4` and `x8`. This implies that our split test set was already exposed to E2E-VarNet during its training phase. Consequently, the results of E2E-VarNet should be considered for reference purposes only and not subjected to rigorous comparison with other models. Both HUMUS-Net(-L) and PromptMR were trained solely on the training set with a `x8` acceleration factor.
</details>
