import itertools
import numpy as np
from torch.utils.data.dataset import Dataset
from .utils import get_normalised_image


''' Utility function for datasets '''


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


''' Datasets '''


class LoadLesionCroppingDataset(Dataset):
    """
    This is a dataset classes, that loads each case in the constructor. The
    idea was to avoid loading all cases in a single step, so instead, what I do
    is load case by case, get its bounding box and keep only the information
    inside it. This is a training dataset and we only want patches that
    actually have lesions since there are lots of non-lesion voxels
    anyways. Just by keeping the bounding box, we use a smaller fraction
    of the RAM used if we loaded all the cases.
    A further improvement in terms of RAM would be to only load the cases
    every time we want a patch. However, that is extremely slow (loading
    NIFTI files, for some reason, is not a fast process). I tried it and
    thought the current approach is the best compromise. We won't need to
    retrain these models anyways ;)
    """
    def __init__(
            self,
            cases, labels, masks, patch_size=32, overlap=0, filtered=True,
            verbose=1
    ):
        # Init
        data_shape = masks[0].shape
        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        if type(overlap) is not tuple:
            overlap = (overlap,) * len(data_shape)

        self.masks = []
        self.labels = []
        self.cases = []
        # That's the big loop that loads all the images. I added some verbosity
        # options for debugging, too. By default they should not be called
        # (verbosity default is 0).
        for i, (mask, case, label) in enumerate(zip(masks, cases, labels)):
            # Indices for the bounding box (mask are already loaded since
            # they are ligh in RAM (binary masks).
            if verbose > 1:
                print(
                    '\033[KIndices {:3d}/{:3d} ({:5.2f})'.format(
                        i, len(cases), 100 * (i + 1) / len(cases)
                    ),
                    end='\r'
                )
            indices = np.where(mask > 0)
            # Bounding box computation.
            if verbose > 1:
                print(
                    '\033[KBB {:3d}/{:3d} ({:5.2f})     '.format(
                        i, len(cases), 100 * (i + 1) / len(cases)
                    ),
                    end='\r'
                )
            bb_i = tuple(
                slice(min_i, max_i)
                for min_i, max_i in zip(
                    np.min(indices, axis=-1), np.max(indices, axis=-1)
                )
            )
            # Mask cropping.
            if verbose > 1:
                print(
                    '\033[KMasks {:3d}/{:3d} ({:5.2f})  '.format(
                        i, len(cases), 100 * (i + 1) / len(cases)
                    ),
                    end='\r'
                )
            self.masks.append(mask[bb_i])
            # Labels (lesion masks) cropping.
            if verbose > 1:
                print(
                    '\033[KLabels {:3d}/{:3d} ({:5.2f}) '.format(
                        i, len(cases), 100 * (i + 1) / len(cases)
                    ),
                    end='\r'
                )
            self.labels.append(label[bb_i])
            # Image loading, normalisation (-mean/std_dev) and cropping.
            if verbose > 1:
                print(
                    '\033[KImages {:3d}/{:3d} ({:5.2f}) '.format(
                        i, len(cases), 100 * (i + 1) / len(cases)
                    ),
                    end='\r'
                )
            self.cases.append(
                np.stack(
                    case,
                    axis=0
                )
            )
            # self.cases.append(
            #     np.stack(
            #         [get_normalised_image(image, mask)[bb_i] for image in case],
            #         axis=0
            #     )
            # )
        if verbose > 1:
            print('\033[K', end='\r')

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.masks, patch_size, overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        if filtered:
            self.patch_slices = [
                [s for s in slices_i if np.sum(label[s]) > 0]
                for label, slices_i in zip(self.labels, slices)
            ]
        else:
            self.patch_slices = slices

        # This cumulative list has two purposes. One of them is giving the
        # length of the dataset ([-1] element), and the other is helping
        # locate the case the patch belongs to given a linear index.
        self.max_slice = np.cumsum(list(map(len, self.patch_slices)))

    def _load_cases(self, cases, masks, bb):
        # Just a function to load all normalised images.
        self.cases = [
            [get_normalised_image(name, mask_i)[bb_i] for name in names_i]
            for names_i, mask_i, bb_i in zip(cases, masks, bb)
        ]

    def __getitem__(self, index):
        # We select the case (here is where max_slice comes in handy ;))
        case_idx = np.min(np.where(self.max_slice > index))
        case = self.cases[case_idx]

        # Slice selector
        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        # Patch "extraction".
        data = case[none_slice + slice_i].astype(np.float32)
        labels = self.labels[case_idx].astype(np.uint8)

        # We expand the labels to have 1 "channel". This is tricky depending
        # on the loss function (some require channels, some don't).
        target = np.expand_dims(labels[slice_i], 0)

        return data, target

    def __len__(self):
        return self.max_slice[-1]
