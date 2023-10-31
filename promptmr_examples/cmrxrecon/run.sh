# 0. prepare dataset
python prepare_h5py_dataset_for_training.py \
--data_path /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil \
--h5py_folder h5_FullSample

# 1. training command
## train promptmr-12cascades model
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_cmrxrecon.py \
--center_numbers 24 \
--accelerations 4 8 10 \
--challenge multicoil \
--mask_type equispaced_fixed \
--data_path /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil \
--h5py_folder h5_FullSample \
--combine_train_val True \
--exp_name promptmr_trainval \
--num_cascades 12 \
--use_checkpoint

## train promptmr-16cascades model
CUDA_VISIBLE_DEVICES=0,1 python train_promptmr_cmrxrecon.py \
--center_numbers 24 \
--accelerations 4 8 10 \
--challenge multicoil \
--mask_type equispaced_fixed \
--data_path /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil \
--h5py_folder h5_FullSample \
--combine_train_val True \
--exp_name promptmr_16_cascades_trainval \
--num_cascades 16 \
--use_checkpoint


# 2. inference for Validation Leaderboard 

## use pretrained promptmr-12cascades model 
CUDA_VISIBLE_DEVICES=1  python run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py \
--input /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil \
--output /research/cbim/vast/bx64/PycharmProjects/cmr_challenge_results/reproduce_promptmr_12_cascades_cmrxrecon \
--model_path pretrained_models/promptmr-12cascades-epoch=11-step=258576.ckpt \
--evaluate_set ValidationSet \
--task Both \
--batch_size 4 \
--num_works 2 \
--center_crop \
--num_cascades 12

## use pretrained promptmr-16cascades model 
CUDA_VISIBLE_DEVICES=1  python run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py \
--input /research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData/MultiCoil \
--output /research/cbim/vast/bx64/PycharmProjects/cmr_challenge_results/reproduce_promptmr_16cascades_cmrxrecon \
--model_path pretrained_models/promptmr-16cascades-epoch=11-step=258576.ckpt \
--evaluate_set ValidationSet \
--task Both \
--batch_size 4 \
--num_works 2 \
--center_crop \
--num_cascades 16





