
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import shutil
import argparse
import numpy as np
from utils import zf_recon
import h5py
import glob
from os.path import join
from tqdm import tqdm

def split_train_val(h5_folder, train_num=100):
    train_folder = join(h5_folder,'train')
    val_folder = join(h5_folder,'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    if os.path.exists(h5_folder):
        num_folders = len(os.listdir(h5_folder))

        for i in range(1, num_folders+1):
            case_folder = join(h5_folder, f"P{i:03d}")
            
            if os.path.exists(case_folder):
                if i<=train_num:
                    shutil.move(case_folder, train_folder)
                else:
                    shutil.move(case_folder, val_folder)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            "--data_path",
            type=str,
            default="/research/cbim/datasets/fastMRI/CMRxRecon/MICCAIChallenge2023/ChallengeData",
            help="Path to the fully sampled cine MATLAB folder",
        )

    parser.add_argument(
        "--save_folder_name",
        type=str,
        default="h5_FullSample",
        help="the folder name to save the h5py files",
    )

    args = parser.parse_args() 
    data_path = args.data_path
    save_folder_name = args.save_folder_name

    fully_cine_matlab_folder = join(data_path, "MultiCoil/Cine/TrainingSet/FullSample")
    fully_mapping_matlab_folder = join(data_path, "MultiCoil/Mapping/TrainingSet/FullSample")

    assert os.path.exists(fully_cine_matlab_folder), f"Path {fully_cine_matlab_folder} does not exist."
    assert os.path.exists(fully_mapping_matlab_folder), f"Path {fully_mapping_matlab_folder} does not exist."

    # 0. get input file list
    f_cine = sorted(glob.glob(join(fully_cine_matlab_folder, '**/*.mat'), recursive=True))
    f_mapping = sorted(glob.glob(join(fully_mapping_matlab_folder, '**/*.mat'), recursive=True))

    f = f_cine + f_mapping
    print('total number of files: ', len(f))
    print('cine cases: ', len(os.listdir(fully_cine_matlab_folder)),' , cine files: ', len(f_cine))
    print('mapping cases: ', len(os.listdir(fully_mapping_matlab_folder)),' , mapping files: ', len(f_mapping))

    # 1. save as fastMRI style h5py files
    for ff in tqdm(f):
        save_path = ff.replace('FullSample',save_folder_name).replace('.mat', '.h5')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        filename = os.path.basename(ff)
        kdata, image = zf_recon(ff)

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset
        save_kdata = kdata.reshape(-1,kdata.shape[2],kdata.shape[3],kdata.shape[4]).transpose(0,1,3,2)
        file.create_dataset('kspace', data=save_kdata)

        file.create_dataset('reconstruction_rss', data=image.reshape(-1,image.shape[3],image.shape[2]))
        file.attrs['max'] = image.max()
        file.attrs['norm'] = np.linalg.norm(image)

        # Add attributes to the dataset
        if 'T1' in filename:
            file.attrs['acquisition'] = 'MOLLI'
        elif 'T2' in filename:
            file.attrs['acquisition'] = 'T2prep-FLASH'
        elif 'lax' in filename:
            file.attrs['acquisition'] = 'TrueFISP-LAX'
        elif 'sax' in filename:
            file.attrs['acquisition'] = 'TrueFISP-SAX'
        else:
            raise ValueError('unknown acquisition type')

        file.attrs['patient_id'] = save_path.split('ChallengeData/')[-1]
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[3]
        file.attrs['encoding_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)
        file.attrs['recon_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)

        # Close the file
        file.close()
    
    # 2. split first 100 cases as training set and the rest 20 cases as validation set
    cine_h5_folder = fully_cine_matlab_folder.replace('FullSample',save_folder_name)
    mapping_h5_folder = fully_mapping_matlab_folder.replace('FullSample',save_folder_name)

    split_train_val(cine_h5_folder, train_num=100)
    split_train_val(mapping_h5_folder, train_num=100)

