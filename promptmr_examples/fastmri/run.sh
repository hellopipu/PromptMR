# split 99 val and 100 test from original multicoil_val set
python split_val_test.py \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_val_origin \
--mask_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_mask

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

# inference using pretrained model
CUDA_VISIBLE_DEVICES=1 python run_pretrained_promptmr_fastmri_knee_inference.py --challenge varnet_knee_mc \
--state_dict_file pretrained_models/promptmr-12cascades-epoch=43-step=764324.ckpt \
--data_path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--output_path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee

# evaluate on test cases with acc=8
python evaluate.py --target-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_full \
--predictions-path /research/cbim/vast/bx64/PycharmProjects/fastmri_results/reproduce_promptmr_knee/reconstructions \
--test-path /research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test \
--challenge multicoil \
--acceleration 8 
