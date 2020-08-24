import time
import itertools
from copy import deepcopy
import numpy as np
from torch.utils.data.dataset import Dataset
from data_manipulation.datasets import get_slices_bb
from data_manipulation.utils import get_normalised_image, time_to_string


def centers_to_slice(voxels, patch_half):
    """
    Function to convert a list of indices defining the center of a patch, to
    a real patch defined using slice objects for each dimension.
    :param voxels: List of indices to the center of the slice.
    :param patch_half: List of integer halves (//) of the patch_size.
    """
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


class BalancedCroppingDataset(Dataset):
    def __init__(
            self,
            cases, labels, masks, patch_size=32, overlap=0, filtered=True,
            balanced=True,
    ):
        # Init
        data_shape = masks[0].shape
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * len(data_shape)
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * len(data_shape)
        else:
            self.overlap = overlap
        self.filtered = filtered
        self.balanced = balanced

        self.cases = []
        self.labels = []
        self.patch_slices = []
        self.bck_slices = []

    def __getitem__(self, index):
        if index < len(self.patch_slices):
            slice_i, case_idx = self.patch_slices[index]
        else:
            index = np.random.randint(len(self.current_bck))
            slice_i, case_idx = self.current_bck.pop(index)
            if len(self.current_bck) == 0:
                self.current_bck = deepcopy(self.bck_slices)

        case = self.cases[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        data = case[none_slice + slice_i].astype(np.float32)
        labels = self.labels[case_idx].astype(np.uint8)

        # We expand the labels to have 1 "channel". This is tricky depending
        # on the loss function (some require channels, some don't).
        target = np.expand_dims(labels[slice_i], 0)

        return data, target

    def __len__(self):
        if self.filtered and self.balanced:
            return len(self.patch_slices) * 2
        else:
            return len(self.patch_slices)


class LesionCroppingDataset(BalancedCroppingDataset):
    """
    This is a training dataset and we only want patches that
    actually have lesions since there are lots of non-lesion voxels
    anyways.
    """
    def __init__(
            self,
            cases, labels, masks, patch_size=32, overlap=0, filtered=True,
            balanced=True,
    ):
        # Init
        super().__init__(
            cases, labels, masks, patch_size, overlap, filtered, balanced
        )

        self.masks = masks
        self.labels = labels
        self.cases = cases

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.masks, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        if self.filtered:
            if self.balanced:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(zip(self.labels, slices))
                    for s in s_i if np.sum(label[s]) > 0
                ]
                self.bck_slices = [
                    (s, i) for i, (label, s_i) in enumerate(zip(self.labels, slices))
                    for s in s_i if np.sum(label[s]) == 0
                ]
                self.current_bck = deepcopy(self.bck_slices)
            else:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(zip(self.labels, slices))
                    for s in s_i if np.sum(label[s]) > 0
                ]
        else:
            self.patch_slices = [
                (s, i) for i, (label, s_i) in enumerate(zip(self.labels, slices))
                for s in s_i
            ]


class DACroppingDataset(Dataset):
    def __init__(
            self,
            source, target, masks_source, masks_target, labels, patch_size=32,
            overlap=0
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap

        self.source = source
        self.target = target
        self.masks_source = masks_source
        self.masks_target = masks_target
        self.labels = labels

        # We get the patch slices (inside the bounding box)...
        s_slices = get_slices(self.masks_source, self.patch_size, self.overlap)
        t_slices = get_slices(self.masks_target, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        self.lesion_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) > 0
        ]
        self.current_lesion = deepcopy(self.lesion_slices)
        self.bck_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) == 0
        ]
        self.current_bck = deepcopy(self.bck_slices)
        self.source_slices = [
            (s, i) for i, s_i in enumerate(s_slices) for s in s_i
        ]

    def __getitem__(self, index):
        # We'll "artificially" balance lesion and background for the target.
        # lesions are a small fraction of the target dataset, but they are
        # the most important voxels, so we want the network to properly
        # model them in the target domain.
        if index % 2 > 0:
            flip = np.random.randint(2)
            if flip:
                t_index = np.random.randint(len(self.lesion_slices))
                t_slice, t_case = self.lesion_slices[t_index]
            else:
                t_index = np.random.randint(len(self.current_lesion))
                t_slice, t_case = self.current_lesion.pop(t_index)
                if len(self.current_lesion) == 0:
                    self.current_lesion = deepcopy(self.lesion_slices)
        else:
            flip = False
            t_index = np.random.randint(len(self.current_bck))
            t_slice, t_case = self.current_bck.pop(t_index)
            if len(self.current_bck) == 0:
                self.current_bck = deepcopy(self.bck_slices)
        s_slice, s_case = self.source_slices[index]

        source = self.source[s_case]
        source_mask = self.masks_source[s_case]
        target = self.target[t_case]
        target_mask = self.masks_target[t_case]
        labels = self.labels[t_case]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        source_data = (
            source[none_slice + s_slice].astype(np.float32),
            np.expand_dims(source_mask[s_slice], axis=0),
        )

        target_data = (
            target[none_slice + t_slice].astype(np.float32),
            np.expand_dims(target_mask[t_slice], axis=0),
            labels[t_slice]
        )
        if flip:
            target_data = (
                np.fliplr(target_data[0]).copy(),
                np.fliplr(target_data[1]).copy(),
                np.flipud(target_data[2]).copy()
            )

        return source_data, target_data

    def __len__(self):
        return len(self.source_slices)


class RLCroppingDataset(Dataset):
    def __init__(
            self,
            source, target, masks_source, masks_target, labels, patch_size=32,
            overlap=0
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap

        self.source = source
        self.target = target
        self.masks_source = masks_source
        self.masks_target = masks_target
        self.labels = labels

        # We get the patch slices (inside the bounding box)...
        self.source_allslices = get_slices(
            self.masks_source, self.patch_size, self.overlap
        )
        t_slices = get_slices(self.masks_target, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        self.lesion_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) > 0
        ]
        self.bck_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) == 0
        ]
        self.current_bck = deepcopy(self.bck_slices)
        self.source_slices = [deepcopy(s) for s in self.source_allslices]
        self.current_source = list(range(len(self.source)))
        # self.curent_source = deepcopy(self.source_slices)

    def __getitem__(self, index):
        # We'll "artificially" balance lesion and background for the target.
        # lesions are a small fraction of the target dataset, but they are
        # the most important voxels, so we want the network to properly
        # model them in the target domain.
        flip = False
        if index < (2 * len(self.lesion_slices)):
            flip = index >= len(self.lesion_slices)
            if flip:
                index -= len(self.lesion_slices)
            t_slice, t_case = self.lesion_slices[index]
        else:
            t_index = np.random.randint(len(self.current_bck))
            t_slice, t_case = self.current_bck.pop(t_index)
            if len(self.current_bck) == 0:
                self.current_bck = deepcopy(self.bck_slices)

        sc_index = np.random.randint(len(self.current_source))
        s_case = self.current_source.pop(sc_index)
        if len(self.current_source) == 0:
            self.current_source = list(range(len(self.source)))
        case_slices = self.source_slices[s_case]
        ss_index = np.random.randint(len(case_slices))
        s_slice = case_slices.pop(ss_index)
        if len(case_slices) == 0:
            self.source_slices[s_case] = deepcopy(
                self.source_allslices[s_case]
            )

        source = self.source[s_case]
        source_mask = self.masks_source[s_case]
        target = self.target[t_case]
        target_mask = self.masks_target[t_case]
        labels = self.labels[t_case]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        source_data = (
            source[none_slice + s_slice].astype(np.float32),
            np.expand_dims(source_mask[s_slice], axis=0),
        )

        target_data = (
            target[none_slice + t_slice].astype(np.float32),
            np.expand_dims(target_mask[t_slice], axis=0),
            labels[t_slice]
        )
        if flip:
            target_data = (
                np.fliplr(target_data[0]).copy(),
                np.fliplr(target_data[1]).copy(),
                np.flipud(target_data[2]).copy()
            )

        return source_data, target_data

    def __len__(self):
        return len(self.lesion_slices) * 4


class DualCroppingDataset(Dataset):
    def __init__(
            self,
            source, target, masks_source, masks_target, labels, patch_size=32,
            overlap=0
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap

        self.source = source
        self.target = target
        self.masks_source = masks_source
        self.masks_target = masks_target
        self.labels = labels

        # We get the patch slices (inside the bounding box)...
        s_slices = get_slices(self.masks_source, self.patch_size, self.overlap)
        t_slices = get_slices(self.masks_target, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        self.lesion_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) > 0
        ]
        self.bck_slices = [
            (s, i) for i, (label, s_i) in enumerate(zip(self.labels, t_slices))
            for s in s_i if np.sum(label[s]) == 0
        ]
        self.current_bck = deepcopy(self.bck_slices)
        self.source_slices = [
            (s, i) for i, s_i in enumerate(s_slices) for s in s_i
        ]

    def __getitem__(self, index):
        flip = False
        if index < len(self.source_slices):
            slice_i, case = self.curent_source[index]
            data = self.source[case]
            labels = np.zeros(data.shape[1:])
            target = False
        else:
            index -= len(self.source_slices)
            # We'll "artificially" balance lesion and background for the target.
            # lesions are a small fraction of the target dataset, but they are
            # the most important voxels, so we want the network to properly
            # model them in the target domain.
            if index < (2 * len(self.lesion_slices)):
                flip = index >= len(self.lesion_slices)
                if flip:
                    index -= len(self.lesion_slices)
                slice_i, case = self.lesion_slices[index]
            else:
                index = np.random.randint(len(self.current_bck))
                slice_i, case = self.current_bck.pop(index)
                if len(self.current_bck) == 0:
                    self.current_bck = deepcopy(self.bck_slices)
            data = self.target[case]
            labels = self.labels[case]
            target = True

        none_slice = (slice(None, None),)
        # Patch "extraction".
        patch = data[none_slice + slice_i].astype(np.float32)
        y = labels[slice_i]
        if flip:
            patch = np.fliplr(patch).copy()
            y = np.fliplr(y).copy()

        return patch, y, target

    def __len__(self):
        return len(self.lesion_slices) * 4 + len(self.source_slices)

