"""
The main file running inside the docker (the starting point)
"""
# Import the required packages
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [2,3])

import numpy as np
import argparse
import time
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



def parse_args():
    """
    Function to control the arguments of the python script when called from the
    command line.
    :return: Dictionary with the argument values
    """
    parser = argparse.ArgumentParser(
        description='Run the lesion identification code based on a '
                    'simple 3D unet.'
    )
    parser.add_argument(
        '-d', '--dataset-path',
        dest='dataset_path',
        default='/home/mariano/SNAC_Lesion_ID_Proj_all',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-o', '--output-path',
        dest='output_path',
        default='/home/mariano/SNAC_Lesion_ID_Proj_all',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=20,
        help='Number of epochs.'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
        help='Patience for early stopping.'
    )
    parser.add_argument(
        '-g', '--gpu',
        dest='gpu_id',
        type=int, default=0,
        help='GPU id number.'
    )
    parser.add_argument(
        '-t', '--task',
        dest='task',
        type=str, default='lit',
        help='GPU id number.'
    )
    parser.add_argument(
        '-m', '--metric_file',
        dest='metric_file',
        type=str, default=None,
        help='GPU id number.'
    )
    parser.add_argument(
        '--run-train',
        dest='run_train',
        action='store_true', default=False,
        help='Whether to train a network on the working directory.'
    )
    parser.add_argument(
        '--run-test',
        dest='run_test',
        action='store_true', default=False,
        help='Whether to test a network on the working directory.'
    )
    parser.add_argument(
        '--general-flag',
        dest='general_flag',
        action='store_true', default=False,
        help='Whether to test a network on the working directory.'
    )
    parser.add_argument(
        '--move-back',
        dest='mov_back',
        action='store_true', default=False,
        help='Whether to move the mask back to original space.'
    )
    return vars(parser.parse_args())


"""
> Main functions (data)
"""


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
    if d_path is None:
        d_path = parse_args()['dataset_path']
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

    if d_path is None:
        d_path = parse_args()['dataset_path']
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
    if d_path is None:
        d_path = parse_args()['dataset_path']
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

    if d_path is None:
        d_path = parse_args()['dataset_path']
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
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if images is None:
        images = ['flair', 't1']
    if im_format is None:
        im_format = '{:}_brain_mni.nii.gz'
    if brain_name is None:
        brain_name = 'flair_brain_mni.nii.gz'

    if task == 'lit':
        brain_name = patient + '_brainmask.nii.gz'
        lesion_name = patient + '_consensus_gt.nii.gz'
    elif task == 'messg':
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
    elif task == 'messg':
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


def remove_small_regions(img_vol, min_size=3):
    """
        Function that removes blobs with a size smaller than a minimum from a mask
        volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param min_size: Minimum size for the blobs.
        :return: New mask without the small blobs.
    """
    # We will need to first define all blobs in the mask. That is 3D connected
    # regions.
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = list(filter(bool, np.unique(blobs)))

    # Finally, for each region we will compute the area and filter those ones
    # that have a smaller area than the threshold.
    areas = [np.count_nonzero(np.equal(blobs, lab)) for lab in labels]
    nu_labels = [lab for lab, a in zip(labels, areas) if a > min_size]
    # We did the filtering in a way that we created a mask for each lesion,
    # so all we need to do is a big union operation of them all.
    nu_mask = reduce(
        lambda x, y: np.logical_or(x, y),
        [np.equal(blobs, lab) for lab in nu_labels]
    ) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


def remove_boundary_regions(img_vol, mask_vol, boundary=2):
    """
    Function to remove regions that are inside the boundary of mask. The
    boundary's thickness can also be defined (default is 2 voxels).
    :param img_vol: Mask volume to process. It should be a numpy array of type
     bool.
    :param mask_vol: Mask from where the boundary area is defined.
    :param boundary: Boundary thickness (default = 2).
    :return: New mask without boundary regions.
    """
    # White matter lesions, should not be on the boundaries of a brain. That
    # region is where the cortex is located which can sometimes present
    # hyperintense artifacts. A way of removing some is to remove all lesions
    # that are on the boundaries (given a certain thickness).

    # First, we'll create a boundary mask by eroding the brain mask and taking
    # only the region of the mask that is not part of the erosion.
    new_mask = np.copy(img_vol)
    im_lab = bwlabeln(img_vol)
    mask_bin = mask_vol.astype(np.bool)
    inner = imerode(
        mask_bin,
        iterations=boundary
    )
    boundary = np.logical_and(mask_bin, np.logical_not(inner))

    # Then it's just a matter of removing any lesion that overlaps with that
    # boundary mask.
    boundary_labs = np.unique(im_lab[boundary])
    boundary_labs = boundary_labs[boundary_labs > 0]
    if len(boundary_labs) > 0:
        new_mask[np.isin(im_lab, boundary_labs)] = 0

    return new_mask


"""
> Main functions (models)
"""


def train_net(
        net, model_name, p_train, patch_size, overlap, images=None,
        batch_size=16, d_path=None, verbose=0, train_val_split=0.2, task=None
):
    """
    Function to train a network with a set of training images.
    :param net: Network to be trained.
    :param model_name: Name that will be used to save the model weights.
    :param patch_size: Size of the patches for training.
    :param overlap: Overlap between the training patches.
    :param batch_size: Number of patches per batch for training.
    :param p_train: Images for training.
    :param d_path: Path to the images.
    :param verbose: Level of verbosity.
    :param train_val_split: Ratio for training and validation set split.
    :param task: Training task name.
    :return: None
    """
    # Init
    c = color_codes()
    if d_path is None:
        d_path = parse_args()['dataset_path']
    epochs = parse_args()['epochs']
    patience = parse_args()['patience']
    training_start = time.time()

    # < UNET >
    try:
        net.load_model(os.path.join(d_path, model_name))
    except IOError:
        if verbose > 0:
            print('Loading the {:}data{:}'.format(c['b'], c['nc']))

        # Mariano's data loading
        # tr_data, tr_lesions, tr_brains = get_data(
        #     p_train, d_path, images=images, preload=False, verbose=verbose
        # )

        if task == 'lit':
            # LIT data loading
            tr_data, tr_lesions, tr_brains = get_lit_data(d_path=d_path)
        elif task == 'msseg':
            # MSSEG2016 data loading
            tr_data, tr_lesions, tr_brains = get_messg_data(d_path=d_path)
        elif task == 'isbi':
            # ISBI data loading
            tr_data, tr_lesions, tr_brains = get_isbi_data(d_path=d_path)
        else:
            # default data loader
            tr_data, tr_lesions, tr_brains = get_data(d_path=d_path)

        # Datasets / Dataloaders should be added here
        if verbose > 1:
            print('Preparing the training datasets / dataloaders')
        # What follows are hard coded parameters. They could (and probably
        # should) be defined as function / command line parameters.
        val_split = train_val_split
        num_workers = 4
        # Here we'll do the training / validation split...
        n_samples = len(tr_data)
        n_t_samples = int(n_samples * (1 - val_split))

        d_train = tr_data[:n_t_samples]
        d_val = tr_data[n_t_samples:]

        l_train = tr_lesions[:n_t_samples]
        l_val = tr_lesions[n_t_samples:]

        m_train = tr_brains[:n_t_samples]
        m_val = tr_brains[n_t_samples:]

        # ... so we can finally define the datasets for the network.
        if verbose > 1:
            print('Training dataset (with validation)')

        train_dataset = LoadLesionCroppingDataset(
            d_train, l_train, m_train, patch_size, overlap,
            verbose=verbose
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers
        )

        if verbose > 1:
            print('Validation dataset')
        val_dataset = LoadLesionCroppingDataset(
            d_val, l_val, m_val, patch_size=patch_size,
            verbose=verbose
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size, num_workers=num_workers
        )

        if verbose > 0:
            n_params = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )
            print(
                '%sStarting training with a unet%s (%s%d%s parameters)' %
                (c['c'], c['nc'], c['b'], n_params, c['nc'])
            )

        # And all that's left is to train and save the model.
        net.fit(
            train_dataloader,
            val_dataloader,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )
        net.save_model(os.path.join(d_path, model_name))

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - training_start)
        )
        print(
            '%sTraining finished%s (total time %s)\n' %
            (c['r'], c['nc'], time_str)
        )


def test_net(
        net, suffix, test_patients, d_path=None, o_path=None, images=None,
        save_pr=True, nii_name='flair_mni.nii.gz', im_name=None,
        brain_name=None, verbose=0, task=None, train_val_split=0.2, portion = 0
):
    """
    Function that tests a fully trained network with a set of testing images.
    :param net: Network fully trained.
    :param suffix: Suffix added to the resulting images.
    :param test_patients: Set of test images.
    :param d_path: Path to the test images.
    :param o_path: Path to store the result images.
    :param images: Image modalities used for testing.
    :param save_pr: Whether to save the probability maps or not.
    :param nii_name: Name of the NIFTI file used for reference when saving the
     results.
    :param im_name: Format for the test image name.
    :param brain_name: Name of the brain mask image.
    :param verbose: Verbosity level
    :return: Lists of the most important metrics and a dictionary with all
     the computed metrics per patient.
    """
    # Init
    c = color_codes()
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if o_path is None:
        o_path = parse_args()['output_path']
    if task is None:
        task = parse_args()['task']
    if images is None:
        images = ['flair', 't1']
    test_start = time.time()
    general_flag = parse_args()['general_flag']
    if parse_args()['metric_file']:
        metric_file = open(os.path.join(o_path, parse_args()['metric_file']), 'w')
    else:
        metric_file = None

    # Here we'll do the training / validation split...
    n_samples = len(test_patients)
    n_t_samples = int(n_samples * (1 - train_val_split))
    test_patients = test_patients[n_t_samples:]

    # Since we are using the network on whole images (not patches if
    # avoidable), we will test image by image. This is extremely fast, so
    # there is no real need to use batches (for testing) of more than one.
    n_test = len(test_patients)
    for pi, p in enumerate(test_patients):
        patient = test_patients[pi]
        patient_path = os.path.join(d_path, patient)
        patient_out_path = os.path.join(o_path, patient)
        if not os.path.isdir(patient_out_path):
            os.mkdir(patient_out_path)
        tests = (n_test - pi + 1)
        if verbose > 0:
            test_elapsed = time.time() - test_start
            test_eta = tests * test_elapsed / (pi + 1)
            print(
                '\033[K{:}Testing with patient {:>13} '
                '{:}({:4d}/{:4d} - {:5.2f}%){:} {:} ETA: {:} {:}'.format(
                    c['g'], patient, c['c'], pi + 1, n_test,
                                             100 * (pi + 1) / n_test, c['g'],
                    time_to_string(test_elapsed), time_to_string(test_eta),
                    c['nc']
                ),
                end='\r'
            )

        # ( Data loading per patient )
        # Load mask, source and target and expand the dimensions.
        testing, tst_brain, gt_lesion_mask, spacing = get_case(
            p, d_path, images=images, im_format=im_name,
            brain_name=brain_name, task=task
        )

        # Brain mask and bounding box
        # Our goal is to only test inside the bounding box of the
        # brain. It doesn't make any sense to look for lesions outside
        # the brain anyways :)
        idx = np.where(tst_brain)

        bb = tuple(
            slice(min_i, max_i)
            for min_i, max_i in zip(
                np.min(idx, axis=-1), np.max(idx, axis=-1)
            )
        )

        # Baseline image (testing)
        # We will us the NIFTI header for the results.
        if task == 'lit':
            nii = load_nii(os.path.join(patient_path, 'flair_corrected.nii.gz'))
        else:
            nii = load_nii(os.path.join(patient_path, nii_name))

        # Here is where we crop the image to the bounding box.
        testing = testing[(slice(None),) + bb]

        # ( Testing itself )
        # That is a tricky part of goof, it was built as a safety measure
        # and it should never be necessary if testing on MNI space.
        # However, I feel better leaving it here, just in case. The idea is
        # to try and test with the bounding box of the whole image (in MNI
        # space). If that fails due to a lack of GPU RAM, we'll catch the
        # exception (which is always a RuntimeError) and we will rerun the
        # test, but this time we'll make patches of the bounding box. The
        # good news is that both testing functions are implemented in the
        # class of the network and they take care of the necessary steps.
        try:
            seg = net.lesions(
                testing, verbose=verbose
            )
        except RuntimeError:
            if verbose > 0:
                test_elapsed = time.time() - test_start
                test_eta = tests * test_elapsed / (pi + 1)
                print(
                    '\033[K{:}CUDA RAM error - '
                    '{:}Testing again with patches patient {:>13} '
                    '{:}({:4d}/{:4d} - {:5.2f}%){:} {:}'
                    ' ETA: {:} {:}'.format(
                        c['r'], c['g'], patient, c['c'],
                        pi + 1, n_test, 100 * (pi + 1) / n_test,
                        c['g'],
                        time_to_string(test_elapsed),
                        time_to_string(test_eta),
                        c['nc']
                    ),
                    end='\r'
                )
            seg = net.patch_lesions(
                testing, patch_size=128,
                verbose=verbose
            )

        # Remember that we cropped the original images, so we need to
        # insert the segmentation into the original position.
        seg_bin = np.zeros_like(tst_brain).astype(np.bool)
        seg_bin[bb] = np.argmax(seg, axis=0).astype(np.bool)

        # This is the only postprocessing currently. The idea is to
        # remove small lesions which are usually false positives.

        lesion_unet = np.logical_and(seg_bin, tst_brain)
        # lesion_unet = remove_boundary_regions(lesion_unet, tst_brain)
        lesion_unet = remove_small_regions(lesion_unet)

        # Finally we'll save a few output images.
        # First we'll save the lesion mask (network output > 0.5).
        mask_nii = nib.Nifti1Image(
            lesion_unet,
            nii.get_qform(),
            nii.get_header()
        )
        mask_nii.to_filename(
            os.path.join(
                patient_out_path, 'lesion_mask_{:}.nii.gz'.format(suffix)
            )
        )

        if general_flag:
            dist = average_surface_distance(gt_lesion_mask, lesion_unet, spacing)
            tpfv = tp_fraction_seg(gt_lesion_mask, lesion_unet)
            fpfv = fp_fraction_seg(gt_lesion_mask, lesion_unet)
            dscv = dsc_seg(gt_lesion_mask, lesion_unet)
            tpfl = tp_fraction_det(gt_lesion_mask, lesion_unet)
            fpfl = fp_fraction_det(gt_lesion_mask, lesion_unet)
            dscl = dsc_det(gt_lesion_mask, lesion_unet)
            tp = true_positive_det(lesion_unet, gt_lesion_mask)
            gt_d = num_regions(gt_lesion_mask)
            lesion_s = num_voxels(lesion_unet)
            gt_s = num_voxels(gt_lesion_mask)
            pdsc = probabilistic_dsc_seg(gt_lesion_mask, lesion_unet)
            if metric_file:
                metric_file.write(
                    '%s;%s;%f;%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n' % (
                        patient+'gt', patient+'pd',
                        dist, tpfv, fpfv, dscv,
                        tpfl, fpfl, dscl,
                        tp, gt_d, lesion_s, gt_s
                    )
                )
            else:
                print(
                    'SurfDist TPFV FPFV DSCV '
                    'TPFL FPFL DSCL '
                    'TPL GTL Voxels GTV PrDSC'
                )
                print(
                    '%f %f %f %f %f %f %f %d %d %d %d %f' % (
                        dist, tpfv, fpfv, dscv,
                        tpfl, fpfl, dscl,
                        tp, gt_d, lesion_s, gt_s, pdsc
                    )
                )
        else:
            sizes = [3, 11, 51]
            tpf, fpf, dscd, dscs = analysis_by_sizes(gt_lesion_mask, lesion_unet, sizes)
            names = '%s;%s;' % (patient+'gt', patient+'pd')
            measures = ';'.join(
                [
                    '%f;%f;%f;%f' % (tpf_i, fpf_i, dscd_i, dscs_i)
                    for tpf_i, fpf_i, dscd_i, dscs_i in zip(
                    tpf, fpf, dscd, dscs
                )
                ]
            )
            if metric_file:
                metric_file.write(names + measures + '\n')
            else:
                intervals = [
                    '\t\t[%d-%d)\t\t|' % (mins, maxs)
                    for mins, maxs in zip(sizes[:-1], sizes[1:])
                ]
                intervals = ''.join(intervals) + \
                            '\t\t[%d-inf)\t|' % sizes[-1]
                measures_s = 'TPF\tFPF\tDSCd\tDSCs\t|' * len(sizes)
                measures = ''.join(
                    [
                        '%.2f\t%.2f\t%.2f\t%.2f\t|' % (
                            tpf_i, fpf_i, dscd_i, dscs_i
                        )
                        for tpf_i, fpf_i, dscd_i, dscs_i in zip(
                        tpf, fpf, dscd, dscs
                    )
                    ]
                )
                print(intervals)
                print(measures_s)
                print(measures)

        # Then we'll save the probability maps. They are complementary
        # because we are dealing with a binary problem. But I just like
        # saving both for visual purposes.
        if save_pr:
            # We'll start with the background / brain probability map...
            pr = np.ones_like(tst_brain).astype(seg.dtype)
            pr[bb] = seg[0]

            seg_nii = nib.Nifti1Image(
                pr,
                nii.get_qform(),
                nii.get_header()
            )
            seg_nii.to_filename(
                os.path.join(patient_out_path, 'back_prob_%s.nii.gz' % suffix)
            )

            # ... and we'll end with the lesion one (which is actually the
            # most interesting one).
            pr = np.zeros_like(tst_brain).astype(seg.dtype)
            pr[bb] = seg[1]
            seg_nii = nib.Nifti1Image(
                pr,
                nii.get_qform(),
                nii.get_header()
            )
            seg_nii.to_filename(
                os.path.join(
                    patient_path, 'lesion_prob_%s.nii.gz' % suffix
                )
            )

    if verbose > 0:
        print(' ' * 100, end='\r')


def train_full_model(
        p_tag='SNAC_WMH',
        d_path=None,
        train_list=None,
        images=None,
        filters=None,
        patch_size=32,
        batch_size=32,
        dropout=.0,
        verbose=1,
        task='lit'
):
    """
    Function to train a model with all the patients of a folder, or the ones
    defined in a specific text file.
    :param p_tag: Tag that must be in the folder name of each patient. By
     default I used the tag that's common on the SNAC cases.
    :param d_path: Path to the images.
    :param train_list: Filename with the lists of patients for training.
    :param images: Images that will be used (default: T1w and FLAIR)
    :param filters: Filters for each layer of the unet.
    :param patch_size: Size of the patches. It can either be a tuple with the
     length of each dimension or just a general length that applies to all
     dimensions.
    :param batch_size: Number of patches per batch. Heavily linked to the
     patch size (memory consumption).
    :param dropout: Dropout value.
    :param verbose: Verbosity level.
    :param task: Task name for model training.
    """
    c = color_codes()

    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if images is None:
        images = ['flair', 't1']
    if filters is None:
        filters = [32, 128, 256, 1024]
    if patch_size is None:
        patch_size = (32, 32, 32)
    if parse_args()['task']:
        task = parse_args()['task']
    try:
        overlap = tuple([p // 2 for p in patch_size])
    except TypeError:
        overlap = (patch_size // 2,) * 3
        patch_size = (patch_size,) * 3
    if train_list is None:
        tmp = get_dirs(d_path)  # if p_tag in p
        p_train = sorted(
            [p for p in tmp],
            key=lambda p: int(''.join(filter(str.isdigit, p)))
        )
    else:
        with open(os.path.join(d_path, train_list)) as f:
            content = f.readlines()
        # We might want to remove whitespace characters like `\n` at the end of
        # each line
        p_train = [x.strip() for x in content]

    if verbose > 0:
        print(
            '{:}[{:}] {:}Training with [{:}] '
            '(filters = [{:}], patch size = [{:}], '
            'dropout = {:4.2f}){:}'.format(
                c['c'], strftime("%H:%M:%S"), c['g'], ', '.join(images),
                ', '.join([str(f) for f in filters]),
                ', '.join([str(ps) for ps in patch_size]),
                dropout, c['nc']
            )
        )
    if verbose > 1:
        print('{:} patients for training'.format(len(p_train)))

    model_name = 'lesions-unet.{:}_model.pt'.format('.'.join(images))
    seg_net = LesionsUNet(
        conv_filters=filters, n_images=len(images), dropout=dropout
    )
    train_net(
        seg_net, model_name, p_train, patch_size, overlap, images=images,
        batch_size=batch_size, d_path=d_path, verbose=verbose, task=task
    )


def test_folder(
        d_path=None, o_path=None, net_name='lesions-unet.{:}_model.pt',
        test_list=None, suffix='unet3d', images=None, filters=None,
        save_pr=False, nii_name='flair_brain_mni.nii.gz', im_name=None,
        brain_name=None, verbose=0, task=None, train_val_split=0.2
):
    """
    Function to test a pretrained model with an unseen dataset.

    :param d_path: Path to the dataset.
    :param o_path: Path to store the result images.
    :param net_name: Name of the file with the model weights.
    :param suffix:  Suffix that will be applied to the final lesion mask.
    :param test_list: Filename with the lists of patients for testing.
    :param images: Images that will be used (default: T1w and FLAIR)
    :param filters: Filters for each layer of the unet.
    :param save_pr: Whether to save the probability maps or not.
    :param nii_name: Name of the NIFTI file used for reference when saving the
     results.
    :param preprocess: Whether to preprocess or not.
    :param im_name: Format for the test image name.
    :param brain_name: Name of the brain mask image.
    :param verbose: Verbosity level
    :return:
    """
    # Init
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if o_path is None:
        o_path = parse_args()['output_path']
    if images is None:
        images = ['flair', 't1']
    if filters is None:
        filters = [32, 128, 256, 1024]
    if task is None:
        task = parse_args()['task']
    if test_list is None:
        patients = sorted(
            get_dirs(d_path), key=lambda p: int(''.join(filter(str.isdigit, p)))
        )
    else:
        with open(os.path.join(d_path, test_list)) as f:
            content = f.readlines()
        # We might want to remove whitespace characters like `\n` at the end of
        # each line
        patients = [x.strip() for x in content]
    # We just have to test the network and save
    # the results.
    seg_net = LesionsUNet(
        conv_filters=filters, n_images=len(images), dropout=0, device='cpu'
    )
    seg_net.load_model(os.path.join(d_path, net_name.format('.'.join(images))))
    test_net(
        seg_net, suffix + '_mni',
        patients, d_path=d_path, o_path=o_path, images=images,
        save_pr=save_pr, nii_name=nii_name, im_name=im_name,
        brain_name=brain_name, verbose=verbose
    )
    # We will also convert all images back to MNI space.
    if parse_args()['mov_back']:
        convert_to_original(
            patients, d_path, o_path, suffix,
            verbose=verbose
        )


def convert_to_original(
        patients, d_path=None, o_path=None, suffix='unet3d',
        fsl_path='/usr/local/fsl/bin', verbose=1
):
    """
    Function to convert MNI images to the original FLAIR space.

    :param patients: List of the patients.
    :param d_path: Path to the dataset.
    :param o_path: Path to store the result images.
    :param suffix: Suffix added to the lesion_mask name.
    :param fsl_path: Path to the FSL library.
    :param verbose: Verbosity level.

    """
    # Init
    c = color_codes()

    # FSL related stuff
    inverse_tool = os.path.join(fsl_path, 'convert_xfm')
    flirt_tool = os.path.join(fsl_path, 'flirt')

    # Patient loading
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if o_path is None:
        o_path = parse_args()['output_path']

    test_start = time.time()
    n_test = len(patients)
    for pi, patient in enumerate(patients):
        patient_path = os.path.join(d_path, patient)
        patient_out_path = os.path.join(o_path, patient)

        if verbose > 0:
            test_elapsed = time.time() - test_start
            test_eta = (n_test - pi + 1) * test_elapsed / (pi + 1)
            print(
                '\033[K{:}Converting patient {:>13} '
                '{:}({:4d}/{:4d} - {:5.2f}%){:} {:} ETA: {:} {:}'.format(
                    c['g'], patient, c['c'], pi + 1, n_test,
                                             100 * (pi + 1) / n_test, c['g'],
                    time_to_string(test_elapsed), time_to_string(test_eta),
                    c['nc']
                ),
                end='\r'
            )

        # Transformation stuff (affine transform, inverse and reference).
        # We will use FSL, but the inverse should be as easy as R^-1 = R'
        # for the rotation and -R^-1 * T for the translation (where R is the
        # rotation part of the matrix, T the translation, R^-1 the inverse of
        # the rotation and R' the transpose of the rotation).
        ref_image = os.path.join(patient_path, 'flair_brain.nii.gz')
        xfm = os.path.join(patient_path, 'flair2mni.mat')
        inv_xfm = os.path.join(patient_out_path, 'mni2flair.mat')
        # Image names
        lesion_mni = os.path.join(
            patient_out_path, 'lesion_mask_{:}_mni.nii.gz'.format(suffix)
        )
        lesion = os.path.join(
            patient_out_path, 'lesion_mask_{:}.nii.gz'.format(suffix)
        )

        # Inverse xfm with convert_xfm
        xfm_cmd = [
            inverse_tool,
            '-omat', inv_xfm,
            '-inverse', xfm
        ]
        check_call(xfm_cmd, stdout=open(os.devnull, 'w'))
        # Mask transformation. We will use nearest neighbour to avoid fuzzy
        # values.
        unet_cmd = [
            flirt_tool,
            '-init', inv_xfm,
            '-interp', 'nearestneighbour',
            '-in', lesion_mni,
            '-ref', ref_image,
            '-applyxfm',
            '-out', lesion
        ]
        check_call(unet_cmd, stdout=open(os.devnull, 'w'))


def main():
    """
    Dummy main function.
    """
    # Training with all cases
    if parse_args()['run_train']:
        train_full_model(verbose=1)
    if parse_args()['run_test']:
        test_folder(verbose=1)
        # test_folder(
        #     net_name='lesions.full-unet.{:}_model.pt', suffix='unet3d.full',
        #     verbose=1
        # )


if __name__ == "__main__":
    main()
