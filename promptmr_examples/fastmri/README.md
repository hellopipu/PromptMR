# Reproducing PromptMR results on fastMRI MultiCoil Knee dataset

## Download the CMRxRecon dataset
Download from fastMRI website: https://fastmri.med.nyu.edu/.

## Preprocess the dataset

Since online evaluation platform is no longer available ([see here](
https://github.com/facebookresearch/fastMRI/discussions/293)), we split 199 validation set into 100 validation set and 99 test set. [We will release this part of code segment along with the mask used in each test case within the upcoming week.]

## Train the model

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_fastmri.py \
--challenge multicoil \
--center_fractions 0.04 \
--accelerations 8 \
--mask_type random \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil \
--combine_train_val False \
--exp_name promptmr_train \
--no_use_ca \
--use_checkpoint
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python run_pretrained_promptmr_fastmri_knee_inference.py --challenge varnet_knee_mc \
--state_dict_file pretrained_models/promptmr-12cascades-epoch=35-step=625356.ckpt \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--output_path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee
```

## Evaluate

```bash
python evaluate.py --target-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_tmp \
--predictions-path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee/reconstructions \
--test-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--challenge multicoil \
--acceleration 8 
```

## Pretrained models
We provide Google Drive links for downloading our models trained on the fastMRI MultiCoil Knee Training Set.


| Model              |# of Params     |Download Link                                                                              |
|--------------------|----------------|-------------------------------------------------------------------------------------------|
| PromptMR-12cascades|80M             |[Link](https://drive.google.com/file/d/1YXgrAoa9MqqSf-6GzXGVkk5-MlQ68w_P/view?usp=sharing) |

## Quantitative results

![fastMRI](../../assets/fastmri_quantitative-poster.png)