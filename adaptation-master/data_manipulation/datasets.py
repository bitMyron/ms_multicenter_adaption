import itertools
from operator import and_
from functools import partial, reduce
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from .generate_features import get_mask_voxels


''' Utility function for datasets '''


def centers_to_slice(voxels, patch_half):
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


def filter_size(slices, mask, min_size):
    filtered_slices = filter(
        lambda s_i: np.sum(mask[s_i] > 0) > min_size, slices
    )

    return list(filtered_slices)


def get_mesh(shape):
    linvec = tuple(np.linspace(0, s - 1, s) for s in shape)
    mesh = np.stack(np.meshgrid(*linvec, indexing='ij')).astype(np.float32)
    return mesh


''' Utility function for patch creation '''


def get_slices_bb(
        masks, patch_size, overlap, rois=None, filtered=False, min_size=0
):
    if rois is None:
        rois = masks
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    if type(masks) is list:
        min_bb = [np.min(np.where(mask > 0), axis=-1) for mask in rois]
        min_bb = [
            [
                min_i + p_len for min_i, p_len in zip(min_bb_i, patch_half)
            ] for min_bb_i in min_bb
        ]
        max_bb = [np.max(np.where(mask > 0), axis=-1) for mask in rois]
        max_bb = [
            [
                max_i - p_len for max_i, p_len in zip(max_bb_i, patch_half)
            ] for max_bb_i in max_bb
        ]

        dim_ranges = [
            map(
                lambda t: np.concatenate([np.arange(*t), [t[1]]]),
                zip(min_bb_i, max_bb_i, steps)
            ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
        ]

        patch_slices = [
            centers_to_slice(
                itertools.product(*dim_range), patch_half
            ) for dim_range in dim_ranges
        ]

        if filtered:
            patch_slices = [
                filter_size(slices, mask, min_size) for slices, mask in zip(
                    patch_slices, masks
                )
            ]

    else:
        # Create bounding box and define
        min_bb = np.min(np.where(rois > 0), axis=-1)
        min_bb = [min_i + p_len for min_i, p_len in zip(min_bb, patch_half)]
        max_bb = np.max(np.where(rois > 0), axis=-1)
        max_bb = [max_i - p_len for max_i, p_len in zip(max_bb, patch_half)]

        dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
        patch_slices = centers_to_slice(
            itertools.product(*dim_range), patch_half
        )

        if filtered:
            patch_slices = filter_size(patch_slices, masks, min_size)

    return patch_slices


def get_balanced_slices(
        masks, patch_size, rois=None, min_size=0,
        neg_ratio=2
):
    # Init
    patch_half = [p_length // 2 for p_length in patch_size]

    # Bounding box + not mask voxels
    if rois is None:
        min_bb = [np.min(np.where(mask > 0), axis=-1) for mask in masks]
        max_bb = [np.max(np.where(mask > 0), axis=-1) for mask in masks]
        bck_masks = map(np.logical_not, masks)
    else:
        min_bb = [np.min(np.where(mask > 0), axis=-1) for mask in rois]
        max_bb = [np.max(np.where(mask > 0), axis=-1) for mask in rois]
        bck_masks = [
            np.logical_and(m, roi.astype(bool)) for m, roi in zip(
                map(np.logical_not, masks), rois
            )
        ]

    # The idea with this is to create a binary representation of illegal
    # positions for possible patches. That means positions that would create
    # patches with a size smaller than patch_size.
    # For notation, i = case; j = dimension
    max_shape = masks[0].shape
    mesh = get_mesh(max_shape)
    legal_masks = [
        reduce(
            np.logical_and,
            [
                np.logical_and(
                    m_j >= max(min_ij, p_ij),
                    m_j <= min(max_ij, max_j - p_ij)
                ) for m_j, min_ij, max_ij, p_ij, max_j in zip(
                    mesh, min_i, max_i, patch_half, max_shape
                )
            ]
        ) for min_i, max_i in zip(min_bb, max_bb)
    ]

    # Filtering with the legal mask
    fmasks = [
        np.logical_and(m, lm) for m, lm in zip(masks, legal_masks)
    ]
    fbck_masks = [
        np.logical_and(m, lm) for m, lm in zip(bck_masks, legal_masks)
    ]

    lesion_voxels = map(get_mask_voxels, fmasks)
    bck_voxels = map(get_mask_voxels, fbck_masks)

    centers_to_halfslice = partial(centers_to_slice, patch_half=patch_half)
    lesion_slices = map(centers_to_halfslice, lesion_voxels)
    bck_slices = map(centers_to_halfslice, bck_voxels)

    # Minimum size filtering for background
    fbck_slices = [
        filter_size(slices, mask, min_size) for slices, mask in zip(
            bck_slices, masks
        )
    ]

    # Final slice selection
    patch_slices = [
        pos_s + [
            neg_s[idx] for idx in np.random.permutation(
                len(neg_s)
            )[:int(neg_ratio * len(pos_s))]
        ] for pos_s, neg_s in zip(lesion_slices, fbck_slices)
    ]

    return patch_slices


''' Datasets '''


# Segmentation (1 timepoint)
class GenericSegmentationCroppingDataset(Dataset):
    def __init__(
            self,
            cases, labels=None, masks=None, balanced=True,
            patch_size=32, neg_ratio=1, sampler=False
    ):
        # Init
        self.neg_ratio = neg_ratio
        # Image and mask should be numpy arrays
        self.sampler = sampler
        self.cases = cases
        self.labels = labels

        self.masks = masks

        data_shape = self.cases[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size

        self.patch_slices = []
        if not self.sampler and balanced:
            if self.masks is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.masks,
                    neg_ratio=self.neg_ratio
                )
            elif self.labels is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.labels,
                    neg_ratio=self.neg_ratio
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(data_single, self.patch_size, 0)
        else:
            overlap = tuple(int(p // 1.1) for p in self.patch_size)
            if self.masks is not None:
                self.patch_slices = get_slices_bb(
                    self.masks, self.patch_size, overlap=overlap,
                    filtered=True
                )
            elif self.labels is not None:
                self.patch_slices = get_slices_bb(
                    self.labels, self.patch_size, overlap=overlap,
                    filtered=True
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] > np.min(d[0]) if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(
                    data_single, self.patch_size, overlap=overlap,
                    filtered=True
                )
        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))
        case = self.cases[case_idx]

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        inputs = case[none_slice + slice_i].astype(np.float32)

        if self.labels is not None:
            labels = self.labels[case_idx].astype(np.uint8)
            target = np.expand_dims(labels[slice_i], 0)

            if self.sampler:
                return inputs, target, index
            else:
                return inputs, target
        else:
            return inputs, case_idx, slice_i

    def __len__(self):
        return self.max_slice[-1]


# Segmentation (2 timepoints)
class LongitudinalCroppingDataset(Dataset):
    def __init__(
            self,
            source, target, lesions, rois=None, patch_size=32, df=True
    ):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = [
            x.shape == y.shape and x.shape[1:] == l.shape for x, y, l in zip(
                source, target, lesions
            )
        ]

        assert reduce(and_, shape_comparisons)

        self.source = source
        self.target = target
        self.lesions = lesions
        self.df = df
        data_shape = self.lesions[0].shape
        self.mesh = get_mesh(data_shape)

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(self.lesions[0].shape)

        self.patch_slices = get_slices_bb(
            lesions, patch_size, tuple(p // 2 for p in patch_size), rois,
            min_size=3
        )
        # self.patch_slices = get_balanced_slices(
        #     lesions, patch_size, rois=rois, neg_ratio=0
        # )

        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

    def __getitem__(self, index):
        # We select the case.
        case = np.min(np.where(self.max_slice > index))
        case_source = self.source[case]
        case_target = self.target[case]
        case_slices = self.patch_slices[case]
        case_lesion = self.lesions[case]

        # Now we just need to look for the desired slice
        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case]
        case_tuple = case_slices[patch_idx]

        # DF's initial mesh to generate a final deformation field.
        none_slice = (slice(None, None),)
        mesh = self.mesh[none_slice + case_tuple]
        source = case_source[none_slice + case_tuple]
        target = case_target[none_slice + case_tuple]
        if self.df:
            inputs_p = (
                source,
                target,
                mesh,
                case_source
            )
            targets_p = (
                np.expand_dims(case_lesion[case_tuple], 0),
                target
            )
        else:
            inputs_p = (
                source,
                target,
            )
            targets_p = np.expand_dims(case_lesion[case_tuple], 0)

        return inputs_p, targets_p

    def __len__(self):
        return self.max_slice[-1]


class LongitudinalImageDataset(Dataset):
    def __init__(
            self,
            source, target, lesions, masks
    ):
        # Init
        # Image and mask should be numpy arrays
        shape_comparisons = [
            x.shape == y.shape and x.shape[1:] == l.shape for x, y, l in zip(
                source, target, lesions
            )
        ]

        assert reduce(and_, shape_comparisons)

        self.source = source
        self.target = target
        self.lesions = lesions
        self.masks = masks

        indices = [np.where(mask > 0) for mask in self.masks]
        self.bb = [
            tuple(
                slice(min_i, max_i)
                for min_i, max_i in zip(
                    np.min(idx, axis=-1), np.max(idx, axis=-1)
                )
            ) for idx in indices
        ]

    def __getitem__(self, index):
        # We select the case.
        bb = self.bb[index]
        source = self.source[index][(slice(None),) + bb]
        target = self.target[index][(slice(None),) + bb]
        lesion = self.lesions[index][bb]

        inputs_p = (
            source,
            target,
        )

        targets_p = (
            np.expand_dims(lesion, 0),
            target
        )

        return inputs_p, targets_p

    def __len__(self):
        return len(self.source)


'''Samplers'''


class WeightedSubsetRandomSampler(Sampler):

    def __init__(self, num_samples, sample_div=2, *args):
        super(WeightedSubsetRandomSampler, self).__init__(args)
        self.total_samples = num_samples
        self.num_samples = num_samples // sample_div
        self.weights = torch.tensor(
            [np.iinfo(np.int16).max] * num_samples, dtype=torch.double
        )
        self.indices = torch.randperm(num_samples)[:self.num_samples]

    def __iter__(self):
        return (i for i in self.indices.tolist())

    def __len__(self):
        return self.num_samples

    def update_weights(self, weights, idx):
        self.weights[idx] = weights.type_as(self.weights)

    def update(self):
        have = 0
        want = self.num_samples // 2
        n_rand = self.num_samples - want
        rand_indices = torch.randperm(self.total_samples)[:n_rand]
        p_ = self.weights.clone()
        p_[rand_indices] = 0
        indices = torch.empty(want, dtype=torch.long)
        while have < want:
            a = torch.multinomial(p_, want - have, replacement=True)
            b = a.unique()
            indices[have:have + b.size(-1)] = b
            p_[b] = 0
            have += b.size(-1)
        self.indices = torch.cat(
            (
                indices[torch.randperm(len(indices))],
                rand_indices
            )
        )
