#!/usr/bin/env python3
import os
from time import strftime
from nibabel import load as load_nii
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import zoom
import argparse
import numpy as np
from operator import add


def main():
    # Parse command line options
    parser = argparse.ArgumentParser(
        description='Test different nets with 3D data.'
    )
    parser.add_argument(
        '-f', '--folder', dest='folder', help="read data from FOLDER",
        default='/home/mariano/DATA/Challenge2016/',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose', default=False
    )
    parser.add_argument(
        '-p', '--patch-size', action='store', type=int, nargs='+',
        dest='patch_size', default=(15, 15, 15)
    )
    parser.add_argument(
        '--use-gado', action='store_true', dest='use_gado'
    )
    parser.add_argument(
        '--no-gado', action='store_false', dest='use_gado', default=False
    )
    parser.add_argument(
        '--gado', action='store', dest='gado',
        default='GADO_preprocessed.nii.gz'
    )
    parser.add_argument(
        '--use-flair', action='store_true', dest='use_flair'
    )
    parser.add_argument(
        '--no-flair', action='store_false', dest='use_flair', default=True
    )
    parser.add_argument(
        '--flair', action='store', dest='flair',
        default='FLAIR_preprocessed.nii.gz'
    )
    parser.add_argument(
        '--use-pd', action='store_true', dest='use_pd'
    )
    parser.add_argument(
        '--no-pd', action='store_false', dest='use_pd', default=True
    )
    parser.add_argument(
        '--pd', action='store', dest='pd',
        default='DP_preprocessed.nii.gz'
    )
    parser.add_argument(
        '--use-t2', action='store_true', dest='use_t2'
    )
    parser.add_argument(
        '--no-t2', action='store_false', dest='use_t2', default=True
    )
    parser.add_argument(
        '--t2', action='store', dest='t2',
        default='T2_preprocessed.nii.gz'
    )
    parser.add_argument(
        '--use-t1', action='store_true', dest='use_t1'
    )
    parser.add_argument(
        '--no-t1', action='store_false', dest='use_t1', default=True
    )
    parser.add_argument(
        '--t1', action='store', dest='t1',
        default='T1_preprocessed.nii.gz'
    )
    parser.add_argument(
        '--mask', action='store', dest='mask', default='Consensus.nii.gz'
    )
    options = vars(parser.parse_args())

    dir_name = options['folder']
    files = sorted(os.listdir(dir_name))
    patients = [f for f in files if os.path.isdir(os.path.join(dir_name, f))]
    n_patients = len(patients)
    for patient, i in zip(patients, range(n_patients)):
        patient_folder = os.path.join(dir_name, patient)
        print(
            '\033[36m[' + strftime("%H:%M:%S") + ']  \033[0mPatient \033[1m' +
            patient + '\033[0m\033[32m (%d/%d)\033[0m' % (i + 1, n_patients)
        )

        mask_nii = load_nii(os.path.join(patient_folder, options['mask']))
        mask_img = mask_nii.get_data()
        lesion_centers = get_mask_voxels(mask_img)

        flair = None
        pd = None
        t1 = None
        t2 = None
        gado = None

        patch_size = tuple(options['patch_size'])
        if options['use_flair']:
            flair = get_patches_from_name(
                os.path.join(patient_folder, options['flair']),
                lesion_centers, patch_size
            )

        if options['use_pd']:
            pd = get_patches_from_name(
                os.path.join(patient_folder, options['pd']),
                lesion_centers, patch_size
            )

        if options['use_t1']:
            t1 = get_patches_from_name(
                os.path.join(patient_folder, options['t1']),
                lesion_centers, patch_size
            )

        if options['use_t2']:
            t2 = get_patches_from_name(
                os.path.join(patient_folder, options['t2']),
                lesion_centers, patch_size
            )

        if options['use_gado']:
            gado = get_patches_from_name(
                os.path.join(patient_folder, options['gado']),
                lesion_centers, patch_size
            )

        patches = np.stack(
            [
                np.array(data)
                for data in [flair, pd, t2, gado, t1] if data is not None
             ],
            axis=1
        )

        print(
                'Our final vector\'s size = ('
                + ','.join([str(num) for num in patches.shape]) + ')'
        )


def get_patches_from_name(filename, centers, patch_size):
    image = load_nii(filename).get_data()
    patches = get_patches(
        image, centers, patch_size
    ) if len(patch_size) == 3 else [
        get_patches2_5d(image, centers, patch_size)
    ]
    return patches


def get_voxels(image, centers):
    return map(lambda center: image[tuple(center)], centers)


def get_patches(image, centers, patch_size=(15, 15, 15), spacing=None):
    # If the size has even numbers, the patch will be centered.
    # If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    list_of_tuples = all(
        map(lambda center: isinstance(center, tuple), centers)
    )
    sizes_match = all(
        map(lambda center: len(center) == len(patch_size), centers)
    )
    if list_of_tuples and sizes_match:
        patch_half = tuple(map(lambda idx: idx / 2, patch_size))
        padding = tuple(
            map(
                lambda idx, size: (idx, size - idx),
                zip(patch_half, patch_size)
            )
        )
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = map(
            lambda center: map(
                lambda c_idx, p_idx, s_idx: slice(
                    c_idx - p_idx, c_idx + (s_idx - p_idx)
                ),
                zip(center, patch_half, patch_size)
            ),
            map(lambda center: map(add, center, patch_half), centers)
        )
        patches = map(lambda idx: new_image[idx], slices)
        if spacing is not None:
            patches = [zoom(patch, spacing) for patch in patches]
    return patches


def get_rolling_patches(image, patch_size):
    """Very basic multi dimensional rolling window. Window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """
    assert type(image) is np.ndarray, 'Image is not a numpy array: %r' % image
    shape = np.array(image.shape * 2)
    strides = np.array(image.strides * 2)
    window = np.asarray(patch_size)
    # new dimensions size
    shape[image.ndim:] = window
    shape[:image.ndim] -= window - 1
    if np.any(shape < 1):
        raise ValueError('window size is too large')
    return np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)


def get_patches2_5d(image, centers, patch_size=(15, 15)):
    # If the size has even numbers, the patch will be centered.
    # If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches_x = []
    patches_y = []
    patches_z = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) >= len(patch_size) for center in centers]
    if list_of_tuples and sizes_match:
        new_patch_size = tuple([max(patch_size)] * len(centers[0]))
        patch_half = tuple([idx / 2 for idx in new_patch_size])
        new_centers = [
            list(map(add, center, patch_half)) for center in centers
        ]
        padding = tuple(
            (idx, size - idx) for idx, size in zip(patch_half, new_patch_size)
        )
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices_x = [
            [
                center[0],
                slice(
                    center[1] - patch_size[0] / 2,
                    center[1] + (patch_size[0] - patch_size[0] / 2)
                ),
                slice(
                    center[2] - patch_size[1] / 2,
                    center[2] + (patch_size[1] - patch_size[1] / 2)
                )
            ]
            for center in new_centers
        ]
        slices_y = [
            [
                slice(
                    center[0] - patch_size[0] / 2,
                    center[0] + (patch_size[0] - patch_size[0] / 2)
                ),
                center[1],
                slice(
                    center[2] - patch_size[1] / 2,
                    center[2] + (patch_size[1] - patch_size[1] / 2)
                )
            ]
            for center in new_centers
        ]
        slices_z = [
            [
                slice(
                    center[0] - patch_size[0] / 2,
                    center[0] + (patch_size[0] - patch_size[0] / 2)
                ),
                slice(
                    center[1] - patch_size[1] / 2,
                    center[1] + (patch_size[1] - patch_size[1] / 2)
                ),
                center[2]
            ]
            for center in new_centers
        ]
        patches_x = [new_image[idx] for idx in slices_x]
        patches_y = [new_image[idx] for idx in slices_y]
        patches_z = [new_image[idx] for idx in slices_z]
    return patches_x, patches_y, patches_z


def get_mask_voxels(mask):
    return map(tuple, np.stack(np.nonzero(mask), axis=1))


def get_mask_centers(mask):
    labels, nlabels = label(mask)
    all_labels = range(1, nlabels + 1)
    centers = map(
        lambda center: tuple(map(int_round, center)),
        center_of_mass(mask, labels, all_labels)
    )
    return centers


def int_round(number):
    return int(round(number))


if __name__ == '__main__':
    main()
