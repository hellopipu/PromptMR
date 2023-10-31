import h5py
import math
import torch
import numpy as np
import torch

############### metric function
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from os.path import join

def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = torch.fft.fftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = torch.fft.fftshift(image_uncentered,dim=dim)
    return image

def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return shape [t,z,y,x]
    '''
    kdata = load_kdata(filename)
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(2)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np

def extract_number(filename):
    '''
    extract number from filename
    '''
    return ''.join(filter(str.isdigit, filename))

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim.item() / gt.shape[0]

def ssim_4d(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 4:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    metric = np.array([0])
    for t_num in range(gt.shape[0]):
        metric = metric + ssim(
            gt[t_num], pred[t_num], maxval=maxval
        )

    return metric.item() / gt.shape[0]

def cal_metric(gt, pred):
    # metric_rmse = mse(gt,pred)**0.5
    metric_nmse = nmse(gt,pred)
    metric_psnr = psnr(gt,pred)
    metric_ssim_4d = ssim_4d(gt,pred)
    # if is_print:
    #     print('mse: {metric_mse:.4f}, nmse: {metric_nmse:.4f}, psnr: {metric_psnr:.4f}, ssim: {metric_ssim_4d:.4f}')
    return metric_nmse, metric_psnr, metric_ssim_4d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )

def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

def load_kdata(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata



############# help[ function #############
def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2), 
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1], indices[3][0]:indices[3][1]]

def crop_submission(a, ismap=False):
    sx,sy,sz,st = a.shape
    if sz>=3:
        a = a[:,:,matlab_round(sz/2)-2:matlab_round(sz/2)]

    if ismap:
        b = _crop(a,(matlab_round(sx/3), matlab_round(sy/2),2,st))
    else:
        b = _crop(a[...,0:3],(matlab_round(sx/3), matlab_round(sy/2),2,3)) 
    return b

# aug func
def rotate(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformationss
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = torch.flip(image,[2])
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = torch.rot90(image, k=2,dims=[2,3])
    elif mode == 5:
        # rotate 180 degree and flip
        out = torch.rot90(image, k=2,dims=[2,3])
        out = torch.flip(out,[2])
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def rotate_re(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = torch.flip(image, [2])
    elif mode == 2:
        # rotate counterwise 90 degree
        out = torch.rot90(image, k=-1)
        # out = np.transpose(image, (1, 0, 2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.flipud(image)
        out = np.rot90(out, k=-1)
    elif mode == 4:
        # rotate 180 degree
        out = torch.rot90(image, k=-2,dims=[2,3])
    elif mode == 5:
        # rotate 180 degree and flip
        out = torch.flip(image,[2])
        out = torch.rot90(out, k=-2, dims=[2,3])
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=-3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=-3)
        
    else:
        raise Exception('Invalid choice of image transformation')

    return out