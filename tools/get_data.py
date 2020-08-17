import os
import numpy as np
import argparse
import time
import torch
from subprocess import check_call
from functools import reduce
from scipy import ndimage as nd
from scipy.ndimage.morphology import binary_erosion as imerode
from nibabel import load as load_nii
import nibabel as nib
from torch.utils.data import DataLoader
from skimage.measure import label as bwlabeln
from time import strftime
from pytorch.utils import get_mask, get_normalised_image
from pytorch.utils import color_codes, get_dirs, print_message, time_to_string
from pytorch.models import LesionsUNet
from pytorch.datasets import LoadLesionCroppingDataset
from data_manipulation.metrics import (
    average_surface_distance, tp_fraction_seg, fp_fraction_seg, dsc_seg,
    tp_fraction_det, fp_fraction_det, dsc_det, true_positive_det, num_regions,
    num_voxels, probabilistic_dsc_seg, analysis_by_sizes
)


def get_data(
        patients,
        d_path=None,
        images=None,
        brain_name='flair_brain_mni.nii.gz',
        lesion_name='lesion_mni.nii.gz',
        preload=True,
        verbose=1,
):
    """

        Function that loads the images and masks of a list of patients.
        :param patients: List of patient names / codes.
        :param d_path: Path to the data.
        :param images: Image names (prefixes).
        :param brain_name: Filename for the brain mask.
        :param lesion_name: Filename for the lesion mask.
        :param preload: Whether to preload the data or not.
        :param verbose: Level of verbosity
        :return: list of numpy arrays for the concatenated images, lesion
        mask and brain mask.
    """
    if images is None:
        images = ['flair', 't1']
    # First we need to find all the patient folders:
    # The assumption is that d_path contains a list of folders, each one with
    # the images of a patient. For this function, however, we already pass the
    # list of patients as a parameter.
    patient_paths = [os.path.join(d_path, p) for p in patients]
    brain_names = [
        os.path.join(p_path, brain_name) for p_path in patient_paths
    ]

    # Brain masks (either a real mask, or an image masked)
    if verbose > 1:
        print('Loading the brain masks')
    brains = list(map(get_mask, brain_names))

    # Lesion masks (we are using this function for training, so there should
    # always be a lesion mask).
    lesion_names = [
        os.path.join(p_path, lesion_name) for p_path in patient_paths
    ]
    if verbose > 1:
        print('Loading the lesion masks')
    lesions = list(map(get_mask, lesion_names))

    # Finally we either load all images and normalise them (a lot of RAM) or we
    # leave that job to the dataset object.
    if verbose > 1:
        print('Loading the images')
    if preload:
        data = [
            np.stack(
                [
                    get_normalised_image(
                        os.path.join(p, '{:}_brain_mni.nii.gz'.format(im)),
                        mask_i,
                    ) for im in images
                ],
                axis=0
            ) for p, mask_i in zip(patient_paths, brains)
        ]
    else:
        data = [
            [
                os.path.join(p, '{:}_brain_mni.nii.gz'.format(im))
                for im in images
            ] for p, mask_i in zip(patient_paths, brains)
        ]

    return data, lesions, brains


def get_lit_data(
        d_path='data/LIT',
        images=None,
        verbose=1,
        preload=True,
):
    """
        Function that loads the images and masks of a list of patients.
        :param d_path: Path to the LSI data.
        :param images: Image names (prefixes).
        :param verbose: Level of verbosity
        :return: list of numpy arrays for the concatenated images, lesion
        mask and brain mask.
    """

    brain_mask_name = 'brainmask.nii.gz'
    lesion_mask_name = 'consensus_gt.nii.gz'
    tmp = get_dirs(d_path)  # if p_tag in p
    p_train = sorted(
        [p for p in tmp],
        key=lambda p: int(''.join(filter(str.isdigit, p)))
    )

    brain_mask_names = [
        os.path.join(d_path, p_path, '_'.join([p_path, brain_mask_name])) for p_path in p_train
    ]

    lesion_names = [
        os.path.join(d_path, p_path, '_'.join([p_path, lesion_mask_name])) for p_path in p_train
    ]
    if images is None:
        images = ['flair', 't1w']
    images = [tmp.upper() for tmp in images]

    # Brain masks (either a real mask, or an image masked)
    if verbose > 1:
        print('Loading the brain masks')
    brains = list(map(get_mask, brain_mask_names))

    # Lesion masks (we are using this function for training, so there should
    # always be a lesion mask).

    if verbose > 1:
        print('Loading the lesion masks')
    lesions = list(map(get_mask, lesion_names))

    # Finally we either load all images and normalise them (a lot of RAM) or we
    # leave that job to the dataset object.
    if verbose > 1:
        print('Loading the images')
    if preload:
        data = [
            np.stack(
                [
                    get_normalised_image(
                        os.path.join(d_path, p, '%s_%s.nii.gz' % (p, im)),
                        mask_i,
                    ) for im in images
                ],
                axis=0
            ) for p, mask_i in zip(p_train, brains)
        ]
    else:
        data = [
            [
                os.path.join(d_path, p, '%s_%s.nii.gz' % (p, im))
                for im in images
            ] for p, mask_i in zip(p_train, brains)
        ]

    return data, lesions, brains

def get_isbi_data(
        d_path='data/ISBI',
        images=None,
        verbose=1,
        preload=True,
):
    """
        Function that loads the images and masks of a list of patients.
        :param d_path: Path to the LSI data.
        :param images: Image names (prefixes).
        :param verbose: Level of verbosity
        :return: list of numpy arrays for the concatenated images, lesion
        mask and brain mask.
    """

    lesion_mask_name = 'mask1.nii'
    tmp = get_dirs(d_path)  # if p_tag in p
    p_train = sorted(
        [p for p in tmp],
        key=lambda p: int(''.join(filter(str.isdigit, p)))
    )
    if images is None:
        images = ['flair', 'mprage']

    # Finally we either load all images and normalise them (a lot of RAM) or we
    # leave that job to the dataset object.
    if verbose > 1:
        print('Loading the images')
    lesion_names = []
    data = []
    for p_path in p_train:
        tmp_stages = set()
        for file in os.listdir(os.path.join(d_path, p_path, 'preprocessed')):
            if file.endswith(".nii"):
                tmp_stages.add(file.split('_')[1])
        for stage in tmp_stages:
            lesion_names.append(os.path.join(d_path, p_path, 'masks', '_'.join([p_path, stage, lesion_mask_name])))
            data.append(np.stack(
                        [
                            get_normalised_image(
                                os.path.join(d_path, p_path, 'preprocessed', '%s_%s_%s_pp.nii' % (p_path, stage, im)),
                                None,
                            ) for im in images
                        ],
                        axis=0
                    ))

    # Lesion masks (we are using this function for training, so there should
    # always be a lesion mask).
    if verbose > 1:
        print('Loading the lesion masks')
    lesions = list(map(get_mask, lesion_names))
    brains = [np.full(lesions[0].shape, 1, dtype=int)]*len(lesions)

    return data, lesions, brains

def get_messg_data(
        d_path='data/LIT',
        images=None,
        verbose=1,
        preload=True,
):
    """
        Function that loads the images and masks of a list of patients.
        :param d_path: Path to the LSI data.
        :param images: Image names (prefixes).
        :param verbose: Level of verbosity
        :return: list of numpy arrays for the concatenated images, lesion
        mask and brain mask.
    """

    brain_mask_name = 'Mask_registered.nii.gz'
    lesion_mask_name = 'Consensus.nii.gz'
    tmp = get_dirs(d_path)  # if p_tag in p
    p_train = sorted(
        [p for p in tmp],
        key=lambda p: int(''.join(filter(str.isdigit, p)))
    )

    brain_mask_names = [
        os.path.join(d_path, p_path, brain_mask_name) for p_path in p_train
    ]

    lesion_names = [
        os.path.join(d_path, p_path, lesion_mask_name) for p_path in p_train
    ]
    if images is None:
        images = ['flair', 't1']
    images = [tmp.upper() for tmp in images]

    # Brain masks (either a real mask, or an image masked)
    if verbose > 1:
        print('Loading the brain masks')
    brains = list(map(get_mask, brain_mask_names))

    # Lesion masks (we are using this function for training, so there should
    # always be a lesion mask).

    if verbose > 1:
        print('Loading the lesion masks')
    lesions = list(map(get_mask, lesion_names))

    # Finally we either load all images and normalise them (a lot of RAM) or we
    # leave that job to the dataset object.
    if verbose > 1:
        print('Loading the images')
    if preload:
        data = [
            np.stack(
                [
                    get_normalised_image(
                        os.path.join(d_path, p, '%s_preprocessed.nii.gz' % im),
                        mask_i,
                    ) for im in images
                ],
                axis=0
            ) for p, mask_i in zip(p_train, brains)
        ]
    else:
        data = [
            [
                os.path.join(d_path, p, '%s_preprocessed.nii.gz' % im)
                for im in images
            ] for p, mask_i in zip(p_train, brains)
        ]

    return data, lesions, brains


def get_case(
        patient,
        d_path=None,
        images=None,
        im_format=None,
        brain_name=None,
        verbose=1,
        task=None
):
    """
        Function that loads the images and masks of a patient.
        :param patient: Patient name / code.
        :param d_path: Path to the data.
        :param images: Image names (prefixes).
        :param im_format: Format for the image name.
        :param brain_name: Filename for the brain mask.
        :param lesion_name: Filename for the lesion mask.
        :param verbose: Level of verbosity
        :return: numpy array for the concatenated images, lesion mask and brain mask.
    """
    if images is None:
        images = ['flair', 't1']
    if im_format is None:
        im_format = '{:}_brain_mni.nii.gz'
    if brain_name is None:
        brain_name = 'flair_brain_mni.nii.gz'

    if task == 'lit':
        brain_name = patient + '_brainmask.nii.gz'
        lesion_name = patient + '_consensus_gt.nii.gz'
    elif task == 'msseg':
        brain_name = 'Mask_registered.nii.gz'
        lesion_name = 'Consensus.nii.gz'
    patient_path = os.path.join(d_path, patient)
    brain_filename = os.path.join(patient_path, brain_name)
    lesion_filename = os.path.join(patient_path, lesion_name)

    # Similarly to the get_data function, here we need to load just one case
    # (for testing). However, we can't assume that we will have a lesion mask,
    # so we will only load the images and a brain mask.

    # Brain mask
    if verbose > 1:
        print('Loading the brain mask')
    brain = get_mask(brain_filename)
    gt_nii = load_nii(lesion_filename)
    lesion = get_mask(lesion_filename)
    spacing = dict(gt_nii.header.items())['pixdim'][1:4]

    # Normalised image
    if verbose > 1:
        print('Loading the images')
    if task == 'lit':
        images = ['FLAIR', 'T1W']
        norm_data = np.stack(
            [
                get_normalised_image(
                    os.path.join(patient_path, '%s_%s.nii.gz' % (patient, im)),
                    brain,
                ) for im in images
            ],
            axis=0
        )
    elif task == 'msseg':
        images = ['FLAIR', 'T1']
        norm_data = np.stack(
            [
                get_normalised_image(
                    os.path.join(patient_path, '%s_preprocessed.nii.gz' % im),
                    brain,
                ) for im in images
            ],
            axis=0
        )
    else:
        norm_data = np.stack(
            [
                get_normalised_image(
                    os.path.join(patient_path, im_format.format(im)),
                    brain,
                ) for im in images
            ],
            axis=0
        )

    return norm_data, brain, lesion, spacing
