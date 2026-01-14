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
import mialab.filtering.feature_extraction as fltr_feat  # kept for structure consistency
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc


atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2

    atlas_t1 = sitk.ReadImage(
        os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz')
    )
    atlas_t2 = sitk.ReadImage(
        os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz')
    )

    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


# -----------------------------------------------------------------------------
# NOTE:
# FeatureExtractor and FeatureImageTypes are intentionally kept
# for STRUCTURE CONSISTENCY, but are NOT USED in the U-Net pipeline.
# -----------------------------------------------------------------------------

class FeatureImageTypes(enum.Enum):
    """Represents the feature image types (unused for U-Net)."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5


# -----------------------------------------------------------------------------
# PRE-PROCESSING
# -----------------------------------------------------------------------------

def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing (skull stripping, normalization)

    NOTE:
    Feature extraction is intentionally REMOVED for U-Net.
    """

    print('-' * 10, 'Processing', id_)

    # load image paths
    path = paths.pop(id_, '')
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')

    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)

    img = structure.BrainImage(id_, path, img, transform)

    # -------------------------------------------------------------------------
    # Brain mask registration (needed for skull stripping)
    # -------------------------------------------------------------------------
    pipeline_brain_mask = fltr.FilterPipeline()

    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(
            fltr_prep.ImageRegistrationParameters(
                atlas_t1, img.transformation, True
            ),
            len(pipeline_brain_mask.filters) - 1
        )

    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask]
    )

    # ensure correct mask type
    img.images[structure.BrainImageTypes.BrainMask] = sitk.Cast(
        img.images[structure.BrainImageTypes.BrainMask],
        sitk.sitkUInt8
    )

    # -------------------------------------------------------------------------
    # T1w pre-processing pipeline
    # -------------------------------------------------------------------------
    pipeline_t1 = fltr.FilterPipeline()

    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(
            fltr_prep.ImageRegistrationParameters(
                atlas_t1, img.transformation
            ),
            len(pipeline_t1.filters) - 1
        )

    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(
            fltr_prep.SkullStrippingParameters(
                img.images[structure.BrainImageTypes.BrainMask]
            ),
            len(pipeline_t1.filters) - 1
        )

    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageNormalization())

    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(
        img.images[structure.BrainImageTypes.T1w]
    )

    # -------------------------------------------------------------------------
    # T2w pre-processing pipeline
    # -------------------------------------------------------------------------
    pipeline_t2 = fltr.FilterPipeline()

    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(
            fltr_prep.ImageRegistrationParameters(
                atlas_t2, img.transformation
            ),
            len(pipeline_t2.filters) - 1
        )

    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(
            fltr_prep.SkullStrippingParameters(
                img.images[structure.BrainImageTypes.BrainMask]
            ),
            len(pipeline_t2.filters) - 1
        )

    if kwargs.get('normalization_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageNormalization())

    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(
        img.images[structure.BrainImageTypes.T2w]
    )

    # -------------------------------------------------------------------------
    # Ground truth registration
    # -------------------------------------------------------------------------
    pipeline_gt = fltr.FilterPipeline()

    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(
            fltr_prep.ImageRegistrationParameters(
                atlas_t1, img.transformation, True
            ),
            len(pipeline_gt.filters) - 1
        )

    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth]
    )

    # update image properties (needed for evaluation)
    img.image_properties = conversion.ImageProperties(
        img.images[structure.BrainImageTypes.T1w]
    )

    return img


# -----------------------------------------------------------------------------
# POST-PROCESSING
# -----------------------------------------------------------------------------

def post_process(img: structure.BrainImage,
                 segmentation: sitk.Image,
                 probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation."""

    print('-' * 10, 'Post-processing', img.id_)

    pipeline = fltr.FilterPipeline()

    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())

    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(
            fltr_postp.DenseCRFParams(
                img.images[structure.BrainImageTypes.T1w],
                img.images[structure.BrainImageTypes.T2w],
                probability
            ),
            len(pipeline.filters) - 1
        )

    return pipeline.execute(segmentation)


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator."""

    def _metric_from(candidates, **kwargs):
        for name in candidates:
            if hasattr(metric, name):
                cls = getattr(metric, name)
                try:
                    return cls(**kwargs) if kwargs else cls()
                except TypeError:
                    continue
        return None

    dice = _metric_from(['DiceCoefficient'])
    jacc = _metric_from(['JaccardCoefficient', 'Jaccard'])
    prec = _metric_from(['Precision', 'PositivePredictiveValue', 'PPV'])
    reca = _metric_from(['Recall', 'Sensitivity', 'TruePositiveRate', 'TPR'])
    hd95 = _metric_from(['HausdorffDistance'], percentile=95)

    metrics = [m for m in [dice, jacc, prec, reca, hd95] if m is not None]

    labels = {
        1: 'WhiteMatter',
        2: 'GreyMatter',
        3: 'Hippocampus',
        4: 'Amygdala',
        5: 'Thalamus'
    }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    return evaluator


# -----------------------------------------------------------------------------
# BATCH PROCESSING
# -----------------------------------------------------------------------------

def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None,
                      multi_process: bool = True):
    """Loads and pre-processes a batch of images."""

    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())

    if multi_process:
        images = mproc.MultiProcessor.run(
            pre_process,
            params_list,
            pre_process_params,
            mproc.PreProcessingPickleHelper
        )
    else:
        images = [
            pre_process(id_, path, **pre_process_params)
            for id_, path in params_list
        ]

    return images


def post_process_batch(brain_images: t.List[structure.BrainImage],
                       segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image],
                       post_process_params: dict = None,
                       multi_process: bool = True):
    """Post-processes a batch of images."""

    if post_process_params is None:
        post_process_params = {}

    param_list = list(zip(brain_images, segmentations, probabilities))

    if multi_process:
        pp_images = mproc.MultiProcessor.run(
            post_process,
            param_list,
            post_process_params,
            mproc.PostProcessingPickleHelper
        )
    else:
        pp_images = [
            post_process(img, seg, prob, **post_process_params)
            for img, seg, prob in param_list
        ]

    return pp_images
