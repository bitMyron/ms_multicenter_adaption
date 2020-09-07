import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [3])

import numpy as np
import time
from subprocess import check_call
from nibabel import load as load_nii
import nibabel as nib
from torch.utils.data import DataLoader
from time import strftime
from pytorch.utils import color_codes, get_dirs, print_message, time_to_string
from pytorch.models import LesionsUNet
from pytorch.datasets import LesionCroppingDataset
from tools.get_data import (
    get_data, get_isbi_data, get_lit_data, get_messg_data, get_case, cross_validation_split, cross_validation_split_isbi
)
from tools.lesion_manipulation import (
    remove_small_regions
)
from tools.lesion_metrics import get_lesion_metrics
from data_manipulation.utils import get_bb
import torch
import itertools
import random

def cross_train_test(
        args, patch_size=32, images=None, filters=None,
        batch_size=16, verbose=0, n_fold=5, task=None
):
    # Init
    c = color_codes()
    dropout = args.get('dropout', 0.5)
    d_path = args.get('dataset_path', None)
    if images is None:
        images = ['flair', 't1']
    if filters is None:
        # filters = [32, 128, 256, 1024]
        # filters = [32, 64, 128, 256]
        filters = [32, 64, 128, 256, 512]
    if patch_size is None:
        patch_size = 32
    # try:
    #     overlap = tuple([p // 2 for p in patch_size])
    # except TypeError:
    #     overlap = (patch_size // 2,) * 3
    #     patch_size = (patch_size,) * 3

    # filters_grid = [
    #                 [32, 64, 128, 256],
    #                 ]

    filters_grid = [
                    [32, 64, 128, 256],
                    [32, 64, 128, 256, 512],
                    # [32, 128, 256, 1024],
                    ]

    # dropout_grid = [0.5]
    dropout_grid = [0,  0.25, 0.5, 0.75]


    if args['task']:
        task = args['task']
    if d_path is None:
        d_path = args['dataset_path']
    o_path = args['output_path']
    if args['metric_file']:
        metric_file = open(os.path.join(o_path, args['metric_file']), 'w')
    else:
        metric_file = None
    grid_search_file = open(os.path.join(o_path, 'grid_result.csv'), 'w')

    epochs = args['epochs']
    patience = args['patience']
    num_workers = 4
    model_name = 'lesions-unet.{:}_model.pt'.format('.'.join(images))
    suffix = 'unet3d'
    training_start = time.time()

    if task == 'lit':
        # LIT data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_lit_data(d_path=d_path)
    elif task == 'msseg':
        # MSSEG2016 data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_messg_data(d_path=d_path)
    elif task == 'isbi':
        # ISBI data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_isbi_data(d_path=d_path)
    else:
        # LIT data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_lit_data(d_path=d_path)

    # Get amount of training samples
    if task == 'isbi':
        cv_indexs = cross_validation_split_isbi(p_trains)
    else:
        cv_indexs = cross_validation_split(len(tr_lesions), n_fold=n_fold)

    spacing = dict(example_nii.header.items())['pixdim'][1:4]

    # Grid search
    for filters, dropout in list(itertools.product(filters_grid, dropout_grid)):

        print("Grid search with: %s;%s;%s\n" % (str(filters), str(dropout), str(patch_size)))

        # Cross train and test
        test_dscs = []
        val_dscs = []
        for i in range(len(cv_indexs)):

            # Save each cv model and test results indexed
            cv_path = os.path.join(o_path, str(i))
            if not os.path.isdir(cv_path):
                os.mkdir(cv_path)

            d_train = [tr_data[tmpi] for tmpi in cv_indexs[i]['train_index']]
            l_train = [tr_lesions[tmpi] for tmpi in cv_indexs[i]['train_index']]
            m_train = [tr_brains[tmpi] for tmpi in cv_indexs[i]['train_index']]

            d_val = [tr_data[tmpi] for tmpi in cv_indexs[i]['val_index']]
            l_val = [tr_lesions[tmpi] for tmpi in cv_indexs[i]['val_index']]
            m_val = [tr_brains[tmpi] for tmpi in cv_indexs[i]['val_index']]

            d_test = [tr_data[tmpi] for tmpi in cv_indexs[i]['test_index']]
            l_test = [tr_lesions[tmpi] for tmpi in cv_indexs[i]['test_index']]
            m_test = [tr_brains[tmpi] for tmpi in cv_indexs[i]['test_index']]
            p_test = [p_trains[tmpi] for tmpi in cv_indexs[i]['test_index']]

            # Initialize unet model
            seg_net = LesionsUNet(
                conv_filters=filters, n_images=len(images), dropout=dropout
            )


            train_dataset = LesionCroppingDataset(
                d_train, l_train, m_train, patch_size, patch_size // 2,
            )
            # Prepare train/val dataloader
            # train_dataset = LoadLesionCroppingDataset(
            #     d_train, l_train, m_train, patch_size, overlap,
            #     verbose=verbose
            # )
            train_dataloader = DataLoader(
                train_dataset, batch_size, True, num_workers=num_workers
            )
            val_dataset = LesionCroppingDataset(
                d_val, l_val, m_val, patch_size * 2, 0, filtered=False,
            )

            val_dataloader = DataLoader(
                val_dataset, batch_size, num_workers=num_workers
            )

            if verbose > 0:
                n_params = sum(
                    p.numel() for p in seg_net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training with a unet%s (%s%d%s parameters)' %
                    (c['c'], c['nc'], c['b'], n_params, c['nc'])
                )

            # And all that's left is to train and save the model.
            seg_net.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs,
                patience=patience,
                verbose=verbose
            )

            for val_case_idx in range(len(d_val)):
                test_brain = d_val[val_case_idx]
                gt_lesion_mask = l_val[val_case_idx]
                try:
                    seg_bb = seg_net.lesions(
                        test_brain, verbose=verbose
                    )
                except RuntimeError:
                    if verbose > 0:
                        print(
                            '\033[K{:}CUDA RAM error - '
                            )
                    seg_bb = seg_net.patch_lesions(
                        test_brain, patch_size=patch_size *2,
                        verbose=verbose
                    )

                # Use thresholding to get binary lesion mask
                # if len(seg_bb.shape) > 3:
                #     seg_im = np.argmax(seg_bb, axis=0) + 1
                # else:
                #     seg_im = seg_bb > 0.5
                seg_im = seg_bb > 0.5
                seg_im = seg_im.astype(int)
                lesion_unet = remove_small_regions(seg_im == 1)

                test_case_dsc = get_lesion_metrics(gt_lesion_mask, lesion_unet, spacing, metric_file, 'val'+str(val_case_idx), fold=i)
                val_dscs.append(test_case_dsc)
                print("%s\n" % str(test_case_dsc))


            for test_case_idx in range(len(p_test)):
                test_brain = d_test[test_case_idx]
                gt_lesion_mask = l_test[test_case_idx]
                # test_brain_mask = m_test[test_case_idx]
                # bb = get_bb(test_brain_mask)

                # seg_im = np.zeros_like(gt_lesion_mask)
                # seg_bb = seg_net.lesions(test_brain)


                try:
                    seg_bb = seg_net.lesions(
                        test_brain, verbose=verbose
                    )
                except RuntimeError:
                    if verbose > 0:
                        print(
                            '\033[K{:}CUDA RAM error - '
                            )
                    seg_bb = seg_net.patch_lesions(
                        test_brain, patch_size=patch_size *2,
                        verbose=verbose
                    )

                if len(seg_bb.shape) > 3:
                    seg_im = np.argmax(seg_bb, axis=0) + 1
                else:
                    seg_im = seg_bb > 0.5
                lesion_unet = remove_small_regions(seg_im == 1)

                # seg_im[np.logical_not(bb)] = 0
                # seg_bin = np.argmax(seg, axis=0).astype(np.bool)
                # # lesion_unet = remove_small_regions(seg_bin)
                # lesion_unet = seg_bin

                mask_nii = nib.Nifti1Image(
                    lesion_unet,
                    example_nii.get_qform(),
                    example_nii.get_header()
                )
                if not os.path.isdir(os.path.join(cv_path, p_test[test_case_idx])):
                    os.mkdir(os.path.join(cv_path, p_test[test_case_idx]))

                mask_nii.to_filename(
                    os.path.join(
                        cv_path, p_test[test_case_idx], 'lesion_mask_{:}.nii.gz'.format(suffix)
                    )
                )

                test_case_dsc = get_lesion_metrics(gt_lesion_mask, lesion_unet, spacing, metric_file, p_test[test_case_idx], fold=i)
                test_dscs.append(test_case_dsc)
                print("%s\n" % str(test_case_dsc))

            seg_net.save_model(os.path.join(cv_path, model_name))

            if verbose > 0:
                time_str = time.strftime(
                    '%H hours %M minutes %S seconds',
                    time.gmtime(time.time() - training_start)
                )
                print(
                    '%sTraining finished%s (total time %s)\n' %
                    (c['r'], c['nc'], time_str)
                )
        grid_search_file.write("%s;%s;%s;%s;%s\n" % (str(filters), str(dropout), str(patch_size),
                                                     str(sum(val_dscs) / len(val_dscs)),
                                                     str(sum(test_dscs)/len(test_dscs))))
        torch.cuda.empty_cache()
    metric_file.close()
    grid_search_file.close()


def train_net(
        args, net, model_name, p_train, patch_size, overlap, images=None, filters=None,
        batch_size=16, d_path=None, verbose=0, train_val_split=0.2, task=None
):
    """
    Function to train a network with a set of training images.
    :param args: arguments from the main funciton.
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
    dropout = args.get('dropout', 0.5)
    d_path = args.get('dataset_path', None)
    if images is None:
        images = ['flair', 't1']
    if filters is None:
        filters = [int(fi) for fi in args.get('filters', '32,64,128,256,512').split(',')]
    if patch_size is None:
        patch_size = 32

    if args['task']:
        task = args['task']
    if d_path is None:
        d_path = args['dataset_path']
    o_path = args['output_path']
    epochs = args['epochs']
    patience = args['patience']
    num_workers = 4
    model_name = 'lesions-unet.{:}_model.pt'.format('.'.join(images))
    suffix = 'unet3d'
    training_start = time.time()

    if task == 'lit':
        # LIT data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_lit_data(d_path=d_path)
    elif task == 'msseg':
        # MSSEG2016 data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_messg_data(d_path=d_path)
    elif task == 'isbi':
        # ISBI data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_isbi_data(d_path=d_path)
    else:
        # LIT data loading
        tr_data, tr_lesions, tr_brains, p_trains, example_nii = get_lit_data(d_path=d_path)

    # Random split the dataset with 90% as training
    dataset_idxes = list(range(len(tr_lesions)))
    random.shuffle(dataset_idxes)
    d_train = [tr_data[tmpi] for tmpi in dataset_idxes[:0.9*len(dataset_idxes)]]
    l_train = [tr_lesions[tmpi] for tmpi in dataset_idxes[:0.9*len(dataset_idxes)]]
    m_train = [tr_brains[tmpi] for tmpi in dataset_idxes[:0.9*len(dataset_idxes)]]
    d_val = [tr_data[tmpi] for tmpi in dataset_idxes[0.9*len(dataset_idxes):]]
    l_val = [tr_lesions[tmpi] for tmpi in dataset_idxes[0.9*len(dataset_idxes):]]
    m_val = [tr_brains[tmpi] for tmpi in dataset_idxes[0.9*len(dataset_idxes):]]

    # Initialize unet model
    seg_net = LesionsUNet(
        conv_filters=filters, n_images=len(images), dropout=dropout
    )
    train_dataset = LesionCroppingDataset(
        d_train, l_train, m_train, patch_size, patch_size // 2,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, True, num_workers=num_workers
    )
    val_dataset = LesionCroppingDataset(
        d_val, l_val, m_val, patch_size * 2, 0, filtered=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size, num_workers=num_workers
    )

    if verbose > 0:
        n_params = sum(
            p.numel() for p in seg_net.parameters() if p.requires_grad
        )
        print(
            '%sStarting training with a unet%s (%s%d%s parameters)' %
            (c['c'], c['nc'], c['b'], n_params, c['nc'])
        )

    # And all that's left is to train and save the model.
    seg_net.fit(
        train_dataloader,
        val_dataloader,
        epochs=epochs,
        patience=patience,
        verbose=verbose
    )

    seg_net.save_model(os.path.join(o_path, model_name))

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - training_start)
        )
        print(
            '%sTraining finished%s (total time %s)\n' %
            (c['r'], c['nc'], time_str)
        )
    torch.cuda.empty_cache()


def test_net(
        args, net, suffix, test_patients, d_path=None, o_path=None, images=None,
        save_pr=True, nii_name='flair_mni.nii.gz', im_name=None,
        brain_name=None, verbose=0, task=None, train_val_split=0.2, portion = 0
):
    """
    Function that tests a fully trained network with a set of testing images.
    :param args: arguments from the main funciton.
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
        d_path = args['dataset_path']
    if o_path is None:
        o_path = args['output_path']
    if task is None:
        task = args['task']
    if images is None:
        images = ['flair', 't1']
    test_start = time.time()
    general_flag = args['general_flag']
    if args['metric_file']:
        metric_file = open(os.path.join(o_path, args['metric_file']), 'w')
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
        elif task == 'msseg':
            nii = load_nii(os.path.join(patient_path, 'FLAIR_preprocessed.nii.gz'))
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

        get_lesion_metrics(gt_lesion_mask, lesion_unet, spacing, metric_file, patient, general_flag)

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
    if metric_file:
        metric_file.close()


# def train_full_model(
#         args,
#         p_tag='SNAC_WMH',
#         d_path=None,
#         train_list=None,
#         images=None,
#         filters=None,
#         patch_size=32,
#         batch_size=32,
#         dropout=.0,
#         verbose=1,
#         task='lit'
# ):
#     """
#     Function to train a model with all the patients of a folder, or the ones
#     defined in a specific text file.
#     :param args: arguments from the main funciton.
#     :param p_tag: Tag that must be in the folder name of each patient. By
#      default I used the tag that's common on the SNAC cases.
#     :param d_path: Path to the images.
#     :param train_list: Filename with the lists of patients for training.
#     :param images: Images that will be used (default: T1w and FLAIR)
#     :param filters: Filters for each layer of the unet.
#     :param patch_size: Size of the patches. It can either be a tuple with the
#      length of each dimension or just a general length that applies to all
#      dimensions.
#     :param batch_size: Number of patches per batch. Heavily linked to the
#      patch size (memory consumption).
#     :param dropout: Dropout value.
#     :param verbose: Verbosity level.
#     :param task: Task name for model training.
#     """
#     c = color_codes()
#
#     # Init
#     if args['dropout']:
#         dropout = args['dropout']
#     if d_path is None:
#         d_path = args['dataset_path']
#     if images is None:
#         images = ['flair', 't1']
#     if filters is None:
#         filters = [32, 128, 256, 1024]
#     if patch_size is None:
#         patch_size = (32, 32, 32)
#     if args['task']:
#         task = args['task']
#     try:
#         overlap = tuple([p // 2 for p in patch_size])
#     except TypeError:
#         overlap = (patch_size // 2,) * 3
#         patch_size = (patch_size,) * 3
#     if train_list is None:
#         tmp = get_dirs(d_path)  # if p_tag in p
#         p_train = sorted(
#             [p for p in tmp],
#             key=lambda p: int(''.join(filter(str.isdigit, p)))
#         )
#     else:
#         with open(os.path.join(d_path, train_list)) as f:
#             content = f.readlines()
#         # We might want to remove whitespace characters like `\n` at the end of
#         # each line
#         p_train = [x.strip() for x in content]
#
#     if verbose > 0:
#         print(
#             '{:}[{:}] {:}Training with [{:}] '
#             '(filters = [{:}], patch size = [{:}], '
#             'dropout = {:4.2f}){:}'.format(
#                 c['c'], strftime("%H:%M:%S"), c['g'], ', '.join(images),
#                 ', '.join([str(f) for f in filters]),
#                 ', '.join([str(ps) for ps in patch_size]),
#                 dropout, c['nc']
#             )
#         )
#     if verbose > 1:
#         print('{:} patients for training'.format(len(p_train)))
#
#     model_name = 'lesions-unet.{:}_model.pt'.format('.'.join(images))
#     seg_net = LesionsUNet(
#         conv_filters=filters, n_images=len(images), dropout=dropout
#     )
#     train_net(
#         seg_net, model_name, p_train, patch_size, overlap, images=images,
#         batch_size=batch_size, d_path=d_path, verbose=verbose, task=task
#     )


def test_folder(
        args, d_path=None, o_path=None, net_name='lesions-unet.{:}_model.pt',
        test_list=None, suffix='unet3d', images=None, filters=None,
        save_pr=False, nii_name='flair_brain_mni.nii.gz', im_name=None,
        brain_name=None, verbose=0, task=None, train_val_split=0.2
):
    """
    Function to test a pretrained model with an unseen dataset.
    :param args: arguments from the main funciton.
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
        d_path = args['dataset_path']
    if o_path is None:
        o_path = args['output_path']
    if images is None:
        images = ['flair', 't1']
    if filters is None:
        filters = [32, 128, 256, 1024]
    if task is None:
        task = args['task']
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
        conv_filters=filters, n_images=len(images), dropout=0
    )
    seg_net.load_model(os.path.join(d_path, net_name.format('.'.join(images))))
    test_net(
        seg_net, suffix + '_mni',
        patients, d_path=d_path, o_path=o_path, images=images,
        save_pr=save_pr, nii_name=nii_name, im_name=im_name,
        brain_name=brain_name, verbose=verbose
    )
    # We will also convert all images back to MNI space.
    if args['mov_back']:
        convert_to_original(
            patients, d_path, o_path, suffix,
            verbose=verbose
        )


def convert_to_original(
        args, patients, d_path=None, o_path=None, suffix='unet3d',
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
        d_path = args['dataset_path']
    if o_path is None:
        o_path = args['output_path']

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