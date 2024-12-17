"""The feature extraction module contains classes for feature extraction."""
import sys

import numpy as np
import pymia.filtering.filter as fltr
import SimpleITK as sitk


class AtlasCoordinates(fltr.Filter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes an atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """
        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # Create matrix with homogeneous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # Reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # Generate transformation matrix
        tmp_mat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmp_mat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features.
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    num_values = len(values)
    p = values / (np.sum(values) + eps)
    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(num_values * (num_values - 1)) / (num_values - 2) * np.sum((values - mean) ** 3) /
                     (num_values * std ** 3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (num_values * std ** 4 + eps),  # kurtosis
                     np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p ** 2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90)
                     ])


class NeighborhoodFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The feature image.

        Raises:
            ValueError: If image is not 3-D.
        """
        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # Test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if not isinstance(function_output, np.ndarray) or function_output.ndim > 1:
            raise ValueError('function must return a 1-D np.ndarray')

        img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])
        img_out_arr = sitk.GetArrayFromImage(img_out)
        img_arr = sitk.GetArrayFromImage(image)

        z, y, x = img_arr.shape
        pad = ((self.kernel[2] // 2, self.kernel[2] // 2),
               (self.kernel[1] // 2, self.kernel[1] // 2),
               (self.kernel[0] // 2, self.kernel[0] // 2))
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')

        for xx in range(x):
            for yy in range(y):
                for zz in range(z):
                    neighborhood = img_arr_padded[zz:zz + self.kernel[2],
                                                  yy:yy + self.kernel[1],
                                                  xx:xx + self.kernel[0]]
                    img_out_arr[zz, yy, xx] = self.function(neighborhood)

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator."""

    @staticmethod
    def get_mask(ground_truth: sitk.Image, ground_truth_labels: list, label_percentages: list,
                 background_mask: sitk.Image = None) -> sitk.Image:
        """Gets a training mask."""
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # Exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x, y, z = indices[no]
                mask_array[x, y, z] = 1  # Mark as masked

        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(ground_truth)
        return mask
