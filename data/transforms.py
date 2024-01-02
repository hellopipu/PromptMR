from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import fastmri

from data.subsample import MaskFunc
from fastmri import fft2c, ifft2c, rss_complex, complex_abs

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies

class PromptMRSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class PromptMrDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        #TODO: cine file does not have left padding
        acq_start = 0
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = PromptMRSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                # attrs=attrs,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = PromptMRSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                # attrs=attrs,
            )

        return sample
    
class FastmriKneePromptMrDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_train_resolution = (384,384)#uniform_train_resolution
    def _crop_if_needed(self, image):
        w_from = h_from = 0
        
        if self.uniform_train_resolution[0] < image.shape[-3]:
            w_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
            w_to = w_from + self.uniform_train_resolution[0]
        else:
            w_to = image.shape[-3]
        
        if self.uniform_train_resolution[1] < image.shape[-2]:
            h_from = (image.shape[-2] - self.uniform_train_resolution[1]) // 2
            h_to = h_from + self.uniform_train_resolution[1]
        else:
            h_to = image.shape[-2]

        return image[..., w_from:w_to, h_from:h_to, :]
    
    def _pad_if_needed(self, image):
        pad_w = self.uniform_train_resolution[0] - image.shape[-3]
        pad_h = self.uniform_train_resolution[1] - image.shape[-2]
        
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
            
        if pad_h > 0:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h - pad_h_left
        else:
            pad_h_left = pad_h_right = 0 
            
        return torch.nn.functional.pad(image.permute(0, 3, 1, 2), (pad_h_left, pad_h_right, pad_w_left, pad_w_right), 'reflect').permute(0, 2, 3, 1)
        
    def _to_uniform_size(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        kspace = fft2c(image)
        return kspace
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> PromptMRSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """

        is_testing = (target is None)

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)

        if not is_testing:
            kspace_torch = self._to_uniform_size(kspace_torch)
        else:
            # Only crop image height
            if self.uniform_train_resolution[0] < kspace_torch.shape[-3]:
                image = ifft2c(kspace_torch)
                h_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
                h_to = h_from + self.uniform_train_resolution[0]
                image = image[..., h_from:h_to, :, :]
                kspace_torch = fft2c(image)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = PromptMRSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = PromptMRSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample