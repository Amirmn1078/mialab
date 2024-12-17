"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings

import numpy as np
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

# Global variables for the atlas images
atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.

    Raises:
        ValueError: If the atlas images do not have the same properties.
    """
    global atlas_t1, atlas_t2

    # Load atlas T1-weighted image
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))

    # Load atlas T2-weighted image
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))

    # Ensure properties match
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images do not have the same image properties')


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""
    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5


class FeatureExtractor:
    """Handles feature extraction for brain images."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The brain image to process.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)

    def execute(self) -> structure.BrainImage:
        """Performs feature extraction on the image.

        Returns:
            structure.BrainImage: The processed image with features.
        """
        # Extract atlas coordinates
        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])

        # Extract T1w intensity and gradient features
        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]

        if self.gradient_intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])

        # Extract T2w intensity and gradient features
        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]

        if self.gradient_intensity_feature:
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])

        # Generate the feature matrix
        self._generate_feature_matrix()

        return self.img

    def _generate_feature_matrix(self):
        """Generates the feature matrix for classification."""
        mask = None

        if self.training:
            # Generate a training mask
            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],  # Labels
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02]  # Sampling percentages
            )

            # Resample the mask to match the image dimensions
            mask = self._resample_mask(mask, self.img.images[structure.BrainImageTypes.T1w])

            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # Generate feature matrix
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for image in self.img.feature_images.values()],
            axis=1
        )

        # Generate labels
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        # Store the feature matrix
        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _resample_mask(mask: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
        """Resamples the mask to match the reference image.

        Args:
            mask (sitk.Image): The mask to resample.
            reference_image (sitk.Image): The reference image.

        Returns:
            sitk.Image: The resampled mask.
        """
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(reference_image)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)
        return resample.Execute(mask)

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Converts an image to a numpy array with optional masking.

        Args:
            image (sitk.Image): The input image.
            mask (np.ndarray): A mask for selecting specific voxels.

        Returns:
            np.ndarray: The flattened array of image features.
        """
        image_arr = sitk.GetArrayFromImage(image)
        components = image.GetNumberOfComponentsPerPixel()

        if mask is not None:
            # Ensure mask and image have the same spatial dimensions
            if mask.shape != image_arr.shape[:3]:
                raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {image_arr.shape[:3]}")

            if components > 1:
                # Expand mask for vector images
                mask = np.repeat(mask[..., np.newaxis], components, axis=3)

            image_arr = np.ma.masked_array(image_arr, mask)

            # Return only unmasked data
            return image_arr[~image_arr.mask].reshape(-1, components)

        # Return all data if no mask is provided
        return image_arr.reshape(-1, components)


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image."""
    print('-' * 10, 'Processing', id_)

    # Load images
    path = paths.pop(id_, '')  # Root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    # Preprocess the brain mask
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(
            fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
            len(pipeline_brain_mask.filters) - 1
        )
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # Preprocess the T1w image
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(
            fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
            len(pipeline_t1.filters) - 1
        )
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(
            fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
            len(pipeline_t1.filters) - 1
        )
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageNormalization())
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

    # Preprocess the T2w image
    pipeline_t2 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(
            fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
            len(pipeline_t2.filters) - 1
        )
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(
            fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
            len(pipeline_t2.filters) - 1
        )
    if kwargs.get('normalization_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageNormalization())
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

    # Preprocess the ground truth
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(
            fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
            len(pipeline_gt.filters) - 1
        )
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # Update image properties
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # Extract features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    # Free up memory
    img.feature_images = {}

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation."""
    print('-' * 10, 'Post-processing', img.id_)

    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator."""
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95)]
    labels = {1: 'WhiteMatter', 2: 'GreyMatter', 3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}
    return eval_.SegmentationEvaluator(metrics, labels)


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, multi_process: bool = True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images."""
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())
    if multi_process:
        return mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        return [pre_process(id_, path, **pre_process_params) for id_, path in params_list]


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process: bool = True) -> t.List[sitk.Image]:
    """Post-processes a batch of images."""
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        return mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                        mproc.PostProcessingPickleHelper)
    else:
        return [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]
