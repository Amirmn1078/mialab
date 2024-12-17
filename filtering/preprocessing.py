"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter using Z-Score normalization with clipping at 5 standard deviations."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes Z-Score normalization with clipping at 5 standard deviations.

        Args:
            image (sitk.Image): The input image to be normalized.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image with outlier clipping.
        """
        img_arr = sitk.GetArrayFromImage(image)

        # Step 1: Calculate mean and standard deviation
        mean = np.mean(img_arr)
        std = np.std(img_arr)

        if std == 0:
            warnings.warn("Standard deviation is zero. Returning the original image.")
            return image

        # Step 2: Z-score normalization
        img_arr = (img_arr - mean) / std

        # Step 3: Clip values at 5 standard deviations
        lower_bound = -3
        upper_bound = 3
        img_arr = np.clip(img_arr, lower_bound, upper_bound)

        # Step 4: Convert back to SimpleITK image
        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization: Z-Score Normalization with Clipping at ±5 Std Dev'


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The skull-stripped image.
        """
        mask = params.img_mask  # the brain mask

        # Remove the skull from the image by using the brain mask
        img_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask)
        stripped_img_arr = img_arr * mask_arr

        img_out = sitk.GetImageFromArray(stripped_img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n'.format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates whether the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """
        # Apply the provided transformation to the image
        transform = params.transformation
        interpolator = sitk.sitkLinear if not params.is_ground_truth else sitk.sitkNearestNeighbor

        registered_image = sitk.Resample(image, params.atlas, transform, interpolator, 0.0, image.GetPixelID())

        return registered_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n'.format(self=self)
