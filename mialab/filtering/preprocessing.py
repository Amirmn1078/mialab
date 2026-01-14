"""
The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities)
for subsequent pipeline steps.
"""

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


# -------------------------------------------------------------------------
# ROBUST INTENSITY CLIPPING (GROUP A – BEST)
# -------------------------------------------------------------------------
class RobustIntensityClipping(pymia_fltr.Filter):
    """Represents a robust intensity clipping filter.

    Intensities are clipped to lower/upper percentiles
    (foreground-only, background assumed to be zero).
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        super().__init__()
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def execute(self, image: sitk.Image,
                params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes robust percentile-based intensity clipping."""

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

        # Foreground mask (after skull stripping)
        fg = img_arr != 0

        if np.any(fg):
            lo = np.percentile(img_arr[fg], self.lower_pct)
            hi = np.percentile(img_arr[fg], self.upper_pct)
            img_arr[fg] = np.clip(img_arr[fg], lo, hi)

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out

    def __str__(self):
        return 'RobustIntensityClipping:\n'


# -------------------------------------------------------------------------
# FOREGROUND Z-SCORE NORMALIZATION (GROUP A – BEST)
# -------------------------------------------------------------------------
class ImageZScoreNormalization(pymia_fltr.Filter):
    """Represents foreground-only Z-score normalization.

    I' = (I - mean_fg) / std_fg
    """

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image,
                params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes Z-score normalization on foreground voxels only."""

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        fg = img_arr != 0

        if np.any(fg):
            mean = img_arr[fg].mean()
            std = img_arr[fg].std()
            if std > 0:
                img_arr[fg] = (img_arr[fg] - mean) / std
            else:
                img_arr[fg] = 0.0

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)
        return img_out

    def __str__(self):
        return 'ImageZScoreNormalization:\n'


# -------------------------------------------------------------------------
# SKULL STRIPPING
# -------------------------------------------------------------------------
class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image,
                params: SkullStrippingParameters = None) -> sitk.Image:
        """Removes the skull using a brain mask."""

        return sitk.Mask(image, params.img_mask)

    def __str__(self):
        return 'SkullStripping:\n'


# -------------------------------------------------------------------------
# IMAGE REGISTRATION
# -------------------------------------------------------------------------
class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self,
                 atlas: sitk.Image,
                 transformation: sitk.Transform,
                 is_ground_truth: bool = False):
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image,
                params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image to the atlas."""

        return sitk.Resample(
            image,
            params.atlas,
            params.transformation,
            sitk.sitkNearestNeighbor if params.is_ground_truth else sitk.sitkLinear,
            0.0,
            image.GetPixelID()
        )

    def __str__(self):
        return 'ImageRegistration:\n'


# -------------------------------------------------------------------------
# BACKWARD COMPATIBILITY ALIAS
# -------------------------------------------------------------------------
# The pipeline expects ImageNormalization().
# We explicitly map it to foreground Z-score normalization.
ImageNormalization = ImageZScoreNormalization
