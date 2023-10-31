import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fastmri.data.subsample import MaskFunc,RandomMaskFunc,EquiSpacedMaskFunc,EquispacedMaskFractionFunc,MagicMaskFunc,MagicMaskFractionFunc

class FixedLowEquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """
    def sample_mask(self,shape,offset):

        num_cols = shape[-2]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies
    
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask
    


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    num_low_frequencies: Optional[int] = None,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fixed":
        return FixedLowEquiSpacedMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    else:
        raise ValueError(f"{mask_type_str} not supported")