import os
from os.path import join
import json
import shutil
import h5py
import argparse

parser = argparse.ArgumentParser(description='Split the original 199 validation set into 99 validation and 100 test set.')
parser.add_argument('--data_path', type=str, default='/research/cbim/datasets/fastMRI/knee_multicoil/multicoil_val_origin', help='path to the original validation set')
parser.add_argument('--mask_path', type=str, default='/research/cbim/datasets/fastMRI/knee_multicoil/multicoil_test_mask', help= 'path to the test mask set')

args = parser.parse_args()

data_path = args.data_path
mask_path = args.mask_path
origin_val_folder = os.path.basename(data_path)
path_prefix = os.path.dirname(data_path) #

val_path = join(path_prefix, 'multicoil_val')
test_path = join(path_prefix, 'multicoil_test')
test_full_path = join(path_prefix, 'multicoil_test_full')

assert os.path.exists(data_path), f'{data_path} does not exist!'
assert os.path.exists(mask_path), f'{mask_path} does not exist!'
assert not os.path.exists(val_path), f'{val_path} already exists!'
assert not os.path.exists(test_path), f'{test_path} already exists!'
assert not os.path.exists(test_full_path), f'{test_full_path} already exists!'

if not os.path.exists(val_path):
    os.makedirs(val_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
if not os.path.exists(test_full_path):
    os.makedirs(test_full_path)

# Read from JSON file
with open('data_split.json', 'r') as f:
    loaded_data = json.load(f)
# Retrieve the split lists
val_file_list = loaded_data['val_list']
test_file_list = loaded_data['test_list']

# copy val set to val_path
for vi in val_file_list:
    vi_origin_path = join(data_path, vi)
    if not os.path.isfile(vi_origin_path):
        continue
    shutil.copy(vi_origin_path, val_path)


# copy test fully-sampled set to test_full_path
# generate masked test set and save to test_path
for ti in test_file_list:
    ti_origin_path = join(data_path, ti)
    ti_mask_path = join(mask_path, ti)
    ti_path = join(test_path, ti)
    if not os.path.isfile(ti_origin_path):
        continue
    # copy test fully-sampled set to test_full_path
    shutil.copy(ti_origin_path, test_full_path)
    with h5py.File(ti_origin_path, 'r') as hf:
        full_kspace = hf['kspace'][()]
    with h5py.File(ti_mask_path, 'r') as hm:
        # generate masked test set and save to test_path
        with h5py.File(ti_path, 'w') as ht:
            masked_kspace = full_kspace*hm['mask'][()][None,None,None,:]+0.0
            ht.create_dataset('kspace', data=masked_kspace)
            ht.create_dataset('ismrmrd_header', data=hm['ismrmrd_header'])
            ht.create_dataset('mask', data=hm['mask'])
            ht.attrs['acquisition'] = hm.attrs['acquisition']
            ht.attrs['patient_id'] = hm.attrs['patient_id']
            ht.attrs['acceleration'] = hm.attrs['acceleration']
            ht.attrs['num_low_frequency'] = hm.attrs['num_low_frequency']  