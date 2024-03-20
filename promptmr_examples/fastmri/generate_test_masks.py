
import os
from os.path import join
import h5py as h5
import json
from tqdm import tqdm
import fastmri
import torch
import xml.etree.ElementTree as etree
from fastmri.data.subsample import create_mask_for_mask_type
import fastmri.data.transforms as T
from fastmri.data.mri_data import et_query

def get_start_end(hf):
    et_root = etree.fromstring(hf["ismrmrd_header"][()])
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_size = (
        int(et_query(et_root, enc + ["x"])),
        int(et_query(et_root, enc + ["y"])),
        int(et_query(et_root, enc + ["z"])),
    )
    rec = ["encoding", "reconSpace", "matrixSize"]
    recon_size = (
        int(et_query(et_root, rec + ["x"])),
        int(et_query(et_root, rec + ["y"])),
        int(et_query(et_root, rec + ["z"])),
    )

    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    enc_limits_center = int(et_query(et_root, lims + ["center"]))
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1
    scanner_model = str(et_query(et_root, ["acquisitionSystemInformation", "systemModel"]))

    padding_left = enc_size[1] // 2 - enc_limits_center
    padding_right = padding_left + enc_limits_max
    return padding_left,padding_right

if __name__ == "__main__":

    # 0. Set the path to the data
    path = '/research/cbim/archive/bx64_extend/datasets/knee_multicoil'
    folder_test_full = 'multicoil_test_full' # folders containing the fully-sampled split test data; can also be multicoil_val_origin
    folder_test_mask = 'multicoil_test_mask'  # folders to save the masks
    path_full = join(path,folder_test_full)
    path_mask = join(path,folder_test_mask)
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)

    # 1. Retrieve the split test list from JSON file
    with open('data_split.json', 'r') as f:
        loaded_data = json.load(f)
    test_file_list = loaded_data['test_list']

    # 2. Generate masks for the test set
    num_4 = 0
    num_8 = 0
    mask_func = create_mask_for_mask_type('random', [0.08, 0.04], [4,8])
    for ff in tqdm(test_file_list):
        fname = join(path_full, ff)
        with h5.File(fname,'r') as hf:
            acq_start, acq_end = get_start_end(hf)
            patient_id = hf.attrs['patient_id']
            print(fname, acq_start, acq_end, patient_id)
            
            seed = tuple(map(ord, ff))
            # kspace = hf['kspace'][()]
            # kspace_torch = T.to_tensor(kspace) #torch.randn((36, 15, 640, 368,2))
            slc,nc,h,w = hf['kspace'].shape
            kspace_torch = torch.randn(slc,nc,h,w,2)

            masked_data, mask, num_low_frequencies = T.apply_mask(kspace_torch, mask_func, seed=seed, padding=(acq_start, acq_end))
            masked_data = T.tensor_to_complex_np(masked_data)
            mask = mask.squeeze().numpy()
            acceleration = mask.sum()/len(mask)
            acceleration = 8 if acceleration<0.18 else 4 # 0.18 is determined by checking the original test set
            if acceleration==4:
                num_4+=1
            else:
                num_8+=1

            with h5.File(fname.replace(folder_test_full,folder_test_mask),'w') as hf2:
                
                hf2.create_dataset('ismrmrd_header', data=hf['ismrmrd_header'])
                # hf2.create_dataset('kspace', data=masked_data)
                hf2.create_dataset('mask', data=mask)
                hf2.attrs['acquisition'] = hf.attrs['acquisition']
                hf2.attrs['patient_id'] = hf.attrs['patient_id']
                hf2.attrs['acceleration'] = acceleration
                hf2.attrs['num_low_frequency'] = num_low_frequencies
        print('acc 4 files: ', num_4, '\n acc 8 files',num_8) 


