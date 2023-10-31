# Reproducing PromptMR results on CMRxRecon dataset

## Download the CMRxRecon dataset
Direct download the [CMRxRecon dataset](https://github.com/CmrxRecon/CMRxRecon) from the following links: 

| Platform           | Link                                                                                                  | Password    |
|--------------------|-------------------------------------------------------------------------------------------------------|-------------|
| Baidu Netdisk      | [Link](https://pan.baidu.com/s/1OXSInGc30gkA4RVYqo9Hqw)                                               | b6hj        |
| Google Drive       | [Link](https://drive.google.com/drive/folders/1--8x5GCnx6Cd2p8ATKLS1bvr3Y0ISvNH?usp=sharing)          | N/A         |

## Preprocess the dataset

For efficient data access during training, it's advisable to convert MATLAB files to fastMRI-compatible h5py format.

```bash
python prepare_h5py_dataset_for_training.py \
--fully_cine_matlab_folder /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample \
--fully_mapping_matlab_folder /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Mapping/TrainingSet/FullSample \
--save_folder_name h5_FullSample
```

## Train the model

```bash
## train promptmr-12cascades model
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_cmrxrecon.py \
--center_numbers 24 \
--accelerations 4 8 10 \
--challenge multicoil \
--mask_type equispaced_fixed \
--data_path /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Cine/TrainingSet/h5_FullSample \
--combine_train_val True \
--exp_name promptmr_trainval \
--use_checkpoint

## train promptmr-16cascades model
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_cmrxrecon.py \
--center_numbers 24 \
--accelerations 4 8 10 \
--challenge multicoil \
--mask_type equispaced_fixed \
--data_path /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil/Cine/TrainingSet/h5_FullSample \
--combine_train_val True \
--exp_name promptmr_16_cascades_trainval \
--num_cascades 16 \
--use_checkpoint
```

## Inference
```bash
## use pretrained promptmr-12cascades model 
CUDA_VISIBLE_DEVICES=1  python run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py \
--input /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData \
--output /research/cbim/vast/bx64/PycharmProjects/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon \
--model_path pretrained_models/promptmr-12cascades-epoch=11-step=258576.ckpt \
--evaluate_set ValidationSet \
--stage 1 \
--batch_size 4 \
--num_works 2 \
--center_crop \
--num_cascades 12

## use pretrained promptmr-16cascades model 
CUDA_VISIBLE_DEVICES=1  python run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py \
--input /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData \
--output /research/cbim/vast/bx64/PycharmProjects/cmr_challenge_results/reproduce_promptmr_16cascades_cmrxrecon \
--model_path pretrained_models/promptmr-16cascades-epoch=11-step=258576.ckpt \
--evaluate_set ValidationSet \
--stage 1 \
--batch_size 4 \
--num_works 2 \
--center_crop \
--num_cascades 16
```

## Pretrained models

We provide Google Drive links for downloading our models trained on the CMRxRecon Training Set (120 cases). You can verify the performance of these models on the [Validation Leaderboard](https://www.synapse.org/#!Synapse:syn51471091/wiki/622548) using the respective IDs. 


| Model              |# of Params     |Download Link                                                                              | Cine Leaderboard ID   | Mapping Leaderboard ID |
|--------------------|----------------|-------------------------------------------------------------------------------------------|-----------------------|------------------------|
| PromptMR-12cascades|82M             |[Link](https://drive.google.com/file/d/1YWMvi1HhC2dC2_hmGJAsfBlOGvZAvYvI/view?usp=sharing) | 9741084               | 9741082                |
| PromptMR-16cascades|108M            |[Link](https://drive.google.com/file/d/1YXB9M9pJ7JY4ld0D3l5a2hAU0UcuyJhN/view?usp=sharing) | 9741143               | 9741142                |

**Note: The leaderboard evaluates only the small central crop area within three slices for each of the 60 validation cases, offering a limited representation of the overall reconstruction results.**

## Quantitative Results
![CMRxRecon Qualitative Results](../../assets/cmrxrecon_qualitative_lax_2ch.png)
![CMRxRecon Qualitative Results](../../assets/cmrxrecon_qualitative_appendix_more.png)

