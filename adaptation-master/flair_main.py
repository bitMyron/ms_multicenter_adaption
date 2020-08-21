import csv
import time
import argparse
import os
from time import strftime
import numpy as np
from functools import reduce
from scipy import ndimage as nd
import nibabel as nib
from nibabel import load as load_nii
from skimage.measure import label as bwlabeln
from torch.utils.data import DataLoader
from data_manipulation.utils import color_codes, get_dirs, get_int
from data_manipulation.utils import get_mask, get_normalised_image
from data_manipulation.utils import time_to_string, get_bb
from data_manipulation.utils import find_file, time_f, print_message
from data_manipulation.utils import save_bland_altman, save_correlation
from data_manipulation.metrics import dsc_seg, tp_fraction_seg, fp_fraction_seg
from data_manipulation.metrics import true_positive_seg
from data_manipulation.metrics import num_regions, num_voxels
from data_manipulation.sitk import itkn4
from models import LesionsUNet
from newmodels import DomainAdapter
from datasets import LesionCroppingDataset
from datasets import RLCroppingDataset, DACroppingDataset

"""
> Arguments
"""


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(
        description='Test different segmentation nets with 3D data.'
    )

    # Mode selector
    parser.add_argument(
        '-v', '--visms-directory',
        dest='visms_dir', default='/home/mariano/data/VISMS_orig',
        help='Path to the VISMS data'
    )
    parser.add_argument(
        '-m', '--msseg-directory',
        dest='msseg_dir', default='/home/mariano/data/MSSEG2016',
        help='Path to the MSSEG 2016 data'
    )
    parser.add_argument(
        '-l', '--lit-directory',
        dest='lit_dir', default='/home/mariano/data/lit/Cross-sectional',
        help='Path to the LIT data'
    )
    parser.add_argument(
        '-w', '--wmh-directory',
        dest='wmh_dir', default='/home/mariano/WMHChallenge2017',
        help='Path to the WMH2017 data'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=20,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=10,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=16,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=32,
        help='Patch size'
    )
    parser.add_argument(
        '--run-test',
        dest='run_test',
        action='store_true', default=False,
        help='Whether to test a network on the working directory.'
    )

    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def get_patient(path, images, brain_name, mask_name, preload):
    roi = get_mask(find_file(brain_name, path))
    mask = get_mask(find_file(mask_name, path))
    if preload:
        im = np.stack(
            [
                get_normalised_image(
                    os.path.join(path, im), roi, masked=True
                ) for im in images
            ], axis=0
        )
    else:
        im = [
            os.path.join(path, im)
            for im in images
        ]
    return im, roi, mask


def get_images(
        d_path, images, patients, brain_name, mask_name, sub_folder='',
        preload=True, verbose=0
):
    """
    Function to get all the images from a folder. For training, files are
    loaded from the loo_dir, which is the one used for a leave-one-out train
    and for testing val_dir is used.
    :param d_path: Path to the images.
    :param images: Image file names.
    :param patients: List of patients
    :param brain_name: Name of the brain mask image.
    :param mask_name: Name of the lesion mask.
    :param sub_folder: Sub folder where the data is stored.
    :param preload: Whether to load the images or not.
    :param verbose: Verbosity level.
    :return:
    """
    c = color_codes()

    patient_dicts = []
    test_start = time.time()

    for pi, p in enumerate(patients):
        p_name = p['name']
        p_dict = {'name': p_name, 't': []}
        p_path = os.path.join(d_path, p_name)
        patient_dicts.append(p_dict)
        tests = len(patients) - pi
        timepoints = p['t']
        if timepoints is not None:
            for t in timepoints:
                if verbose > 0:
                    test_elapsed = time.time() - test_start
                    test_eta = tests * test_elapsed / (pi + 1)
                    print(
                        '{:}Loading patient {:} ({:d}/{:d}) [t: {:}] '
                        '{:} ETA {:}'.format(
                            c['clr'], p_name, pi + 1, len(patients), t,
                            time_to_string(test_elapsed),
                            time_to_string(test_eta),
                        ), end='\r'
                    )
                t_path = os.path.join(p_path, t, sub_folder)
                im, roi, mask = get_patient(
                    t_path, images, brain_name, mask_name, preload
                )
                t_dict = {
                    'name': t,
                    'roi': roi,
                    'images': im,
                    'mask': mask,
                }
                p_dict['t'].append(t_dict)
        else:
            if verbose > 0:
                test_elapsed = time.time() - test_start
                test_eta = tests * test_elapsed / (pi + 1)
                print(
                    '{:}Loading patient {:} ({:d}/{:d}) {:} ETA {:}'.format(
                        c['clr'], p_name, pi + 1, len(patients),
                        time_to_string(test_elapsed),
                        time_to_string(test_eta),
                    ), end='\r'
                )
            sub_path = os.path.join(p_path, sub_folder)
            im, roi, mask = get_patient(
                sub_path, images, brain_name, mask_name, preload
            )
            t_dict = {
                'name': 'Unknown',
                'roi': roi,
                'images': im,
                'mask': mask,
            }
            p_dict['t'] = [t_dict]

    if verbose > 0:
        test_elapsed = time.time() - test_start
        print(
            '{:}All patients loaded {:}'.format(
                c['clr'], time_to_string(test_elapsed)
            )
        )

    return patient_dicts


def prepare_visms_patients(d_path, verbose=0):
    images = ['flair_corrected.nii.gz']
    patients = sorted(get_dirs(d_path), key=get_int)
    patient_dicts = [
        {
            'name': p, 't': get_dirs(os.path.join(d_path, p))
        }
        for p in patients
    ]

    brain_name = 'brain_mask.nii.gz'
    mask_name = 'lesion_mask.nii.gz'

    images = get_images(
        d_path=d_path, patients=patient_dicts, images=images,
        brain_name=brain_name, mask_name=mask_name, verbose=verbose
    )

    return images


def prepare_msseg_patients(d_path, verbose=0):
    images = ['FLAIR_preprocessed.nii.gz']
    patients = sorted(get_dirs(d_path), key=get_int)
    patient_dicts = [
        {
            'name': p, 't': None
        }
        for p in patients
    ]

    brain_name = 'Mask_registered.nii.gz'
    mask_name = 'Consensus.nii.gz'

    images = get_images(
        d_path=d_path, patients=patient_dicts, images=images,
        brain_name=brain_name, mask_name=mask_name, verbose=verbose
    )

    return images


def prepare_lit_patients(d_path, verbose=0):
    images = ['flair_corrected.nii.gz']
    patients = sorted(get_dirs(d_path), key=get_int)

    for p in patients:
        p_path = os.path.join(d_path, p)
        if verbose > 1:
            print_message('- N4 (FLAIR) {:}'.format(p))
        filename = os.path.join(p_path, p + '_FLAIR.nii.gz')
        time_f(lambda: itkn4(filename, p_path, 'flair'))

    patient_dicts = [
        {
            'name': p, 't': None
        }
        for p in patients
    ]

    brain_name = 'brainmask.nii.gz'
    mask_name = 'consensus_gt.nii.gz'

    images = get_images(
        d_path=d_path, patients=patient_dicts, images=images,
        brain_name=brain_name, mask_name=mask_name, verbose=verbose
    )

    return images


def remove_small_regions(img_vol, min_size=3):
    """
        Function that removes blobs with a size smaller than a minimum from a mask
        volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param min_size: Minimum size for the blobs.
        :return: New mask without the small blobs.
    """
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = list(filter(bool, np.unique(blobs)))
    areas = [np.count_nonzero(np.equal(blobs, lab)) for lab in labels]
    nu_labels = [lab for lab, a in zip(labels, areas) if a > min_size]
    nu_mask = reduce(
        lambda x, y: np.logical_or(x, y),
        [np.equal(blobs, lab) for lab in nu_labels]
    ) if nu_labels else np.zeros_like(img_vol)
    return nu_mask


"""
> Evaluation functions
"""


def patient_analysis(gt, lesions, p_name):
    patient_dict = {
        'name': p_name,
    }

    # ( Segmentation evaluation )
    # We want to get a few measures per case and then compute the
    # means.
    # It's important to keep in mind that the experiments are NOT
    # comparable. Even though we are using the same number of testing
    # cases, they are different subsets.
    gt_bool = gt.astype(np.bool)
    gt_lab = bwlabeln(gt_bool)

    # Pretty common and normal voxelwise / segmentation metrics.
    patient_dict['tpfv'] = tp_fraction_seg(gt, lesions)
    patient_dict['fpfv'] = fp_fraction_seg(gt, lesions)
    patient_dict['dscv'] = dsc_seg(gt, lesions)
    patient_dict['tpv'] = true_positive_seg(gt, lesions)

    # Some intermediate steps.
    auto_bool = lesions.astype(np.bool)
    auto_lab = bwlabeln(lesions.astype(np.bool))
    auto_labs = np.unique(auto_lab)

    # And finally a few regionwise / detection metrics.
    tp_labs = np.unique(gt_lab[auto_bool])
    notfp_labs = np.unique(auto_lab[gt_bool])
    fp_mask = np.logical_not(np.isin(auto_lab, notfp_labs))
    fp_labs = np.unique(auto_lab[fp_mask])
    tp = len(tp_labs[tp_labs > 0])
    patient_dict['tp'] = len(tp_labs[tp_labs > 0])
    fp = len(fp_labs[fp_labs > 0])
    gt_d = num_regions(gt)
    patient_dict['gt_d'] = gt_d
    patient_dict['tpfl'] = 100 * tp / gt_d if gt_d > 0 else 0
    d = len(auto_labs[auto_labs > 0])
    patient_dict['fpfl'] = 100 * fp / d if d > 0 else 0
    patient_dict['lesion_s'] = num_voxels(lesions)
    patient_dict['gt_s'] = num_voxels(gt)

    return patient_dict


def general_analysis(net, data, suffix='baseline', model_name=None, verbose=0):
    """
        Function to perform a general analysis on the masks obtained during
        testing. Includes correlation estimates.
        :param net: Pretrained network to be evaluated.
        :param data: Data for testing.
        :param suffix: Suffix added to the resulting images.
        :param model_name: General filename (used only for DA nets).
        :param verbose: Verbosity level.
        :return: Dictionary with all the lesion details.
        """
    # Init
    c = color_codes()
    d_path = parse_inputs()['visms_dir']

    dsc_list = list()
    tpf_list = list()
    fpf_list = list()
    test_dicts = list()

    test_start = time.time()
    n_test = len(data)

    for pi, (p_les, t_les) in enumerate(data):
        if verbose > 1:
            t_elapsed = time.time() - test_start
            elapsed_s = time_to_string(t_elapsed)
            if pi > 0:
                steps_left = 2 * (n_test - pi) / (2 * pi)
                eta_s = time_to_string(steps_left * t_elapsed)
            else:
                eta_s = ''
            print(
                '{:}Testing patient    {:<15} (t: {:}) '
                '{:}[{:3d}/{:3d} - {:5.2f}%] {:} ETA: {:}'.format(
                    c['clr'], c['g'] + p_les['name'] + c['nc'],
                              c['c'] + t_les['name'] + c['nc'],
                    c['c'], pi + 1, n_test, 100 * (pi + 1) / n_test,
                              c['g'] + elapsed_s, eta_s + c['nc']
                ),
                end='\r'
            )

        # Reference data
        gt = t_les['mask']

        # Mask loading
        bb = get_bb(t_les['roi'])
        image = t_les['images'][(slice(None),) + bb]
        # Testing
        if model_name is not None:
            model_path = os.path.join(d_path, model_name.format(p_les['name']))
            net.load_model(model_path)
        lesions = test(net, image, t_les['roi'])

        if verbose > 0:
            t_elapsed = time.time() - test_start
            elapsed_s = time_to_string(t_elapsed)
            steps_left = (2 * (n_test - pi) - 1) / (2 * pi + 1)
            eta_s = time_to_string(steps_left * t_elapsed)
            print(
                '{:}Evaluating patient {:<15} (t: {:}) '
                '{:}[{:3d}/{:3d} - {:5.2f}%] {:} ETA: {:}'.format(
                    c['clr'], c['g'] + p_les['name'] + c['nc'],
                              c['c'] + t_les['name'] + c['nc'],
                    c['c'], pi + 1, n_test, 100 * (pi + 1) / n_test,
                              c['g'] + elapsed_s, eta_s + c['nc']
                ),
                end='\r'
            )

        patient_dict = patient_analysis(gt, lesions, p_les['name'])

        if patient_dict['gt_d'] > 0:
            tpf_list.append(patient_dict['tpfl'])
            fpf_list.append(patient_dict['fpfl'])
            dsc_list.append(patient_dict['dscv'])
        test_dicts.append(patient_dict)

    tp_vol = np.array([d['tpv'] for d in test_dicts])
    x_vol = np.array([d['lesion_s'] for d in test_dicts])
    y_vol = np.array([d['gt_s'] for d in test_dicts])

    xy_valid = np.logical_and(x_vol > 0, y_vol > 0)
    x_log = np.log(x_vol[xy_valid])
    y_log = np.log(y_vol[xy_valid])
    save_bland_altman(x_vol, y_vol, suffix, d_path)
    save_bland_altman(x_log, y_log, 'log.' + suffix, d_path)
    r, s, t = save_correlation(x_vol, y_vol, suffix, d_path)
    log_r, _, _ = save_correlation(x_log, y_log, 'log.' + suffix, d_path)

    global_metrics = {
        'tpfl': np.mean(tpf_list),
        'fpfl': np.mean(fpf_list),
        'dscv': np.mean(dsc_list),
        'dsc_global': 2 * np.sum(tp_vol) / (np.sum(x_vol) + np.sum(y_vol)),
        'r': r,
        'log_r': log_r,
        's': s,
        't': t,
    }

    with open(
            os.path.join(d_path, 'results_{:}.csv'.format(suffix)), 'w'
    ) as csvfile:
        evalwriter = csv.writer(csvfile)
        evalwriter.writerow(
            [
                'Patient', 'TPFV', 'FPFV', 'DSCV', 'TPFL', 'FPFL',
                'TPL', 'GTL', 'Vox', 'GTV'
            ]
        )
        for d in test_dicts:
            evalwriter.writerow(
                [
                    d['name'], d['tpfv'], d['fpfv'], d['dscv'],
                    d['tpfl'], d['fpfl'],
                    d['tp'], d['gt_d'], d['lesion_s'], d['gt_s'],
                ]
            )

    if verbose > 0:
        print(
            '{:}{:^15}|{:6.2f}|{:6.2f}|{:6.4f}|{:6.4f}|'
            '|{:6.4f}|{:6.4f}|{:6.4f}|{:6.4f}|'.format(
                c['clr'], suffix,
                global_metrics['tpfl'], global_metrics['fpfl'],
                global_metrics['dscv'], global_metrics['dsc_global'],
                global_metrics['r'], global_metrics['log_r'],
                global_metrics['s'], global_metrics['t']
            )
        )


"""
> Network functions
"""


def train(
        net, model_name, patient_dicts, val_split=0.1, model_type='Unet',
        source=None, verbose=0
):
    """

    :param net:
    :param model_name:
    :param patient_dicts:
    :param val_split:
    :param model_type:
    :param source:
    :param verbose:
    :return:
    """
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    patience = options['patience']
    batch_size = options['batch_size']
    patch_size = options['patch_size']
    d_path = options['visms_dir']

    try:
        net.load_model(os.path.join(d_path, model_name))
    except IOError:

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        # Here we'll do the training / validation split...
        training = [t for p in patient_dicts for t in p['t']]
        data = [t['images'] for t in training]
        rois = [t['roi'] for t in training]
        mask = [t['mask'] for t in training]

        n_samples = len(training)
        n_val = int(n_samples * val_split)

        if val_split > 0:
            d_train = data[n_val:]
            d_val = data[:n_val]
            m_train = mask[n_val:]
            m_val = mask[:n_val]
            r_train = rois[n_val:]
            r_val = rois[:n_val]
        else:
            d_val = d_train = data
            m_val = m_train = mask
            r_val = r_train = rois

        if source is None:
            train_dataset = LesionCroppingDataset(
                d_train, m_train, r_train, patch_size, patch_size // 2,
            )
        else:
            patience = epochs
            ns_samples = len(source)
            ns_val = int(ns_samples * val_split)
            s_data = [t['images'] for _, t in source]
            s_rois = [t['roi'] for _, t in source]
            if val_split > 0:
                sd_train = s_data[ns_val:]
                sr_train = s_rois[ns_val:]
                sd_val = s_data[:ns_val]
                sr_val = s_rois[:ns_val]
            else:
                sd_val = sd_train = s_data
                sr_val = sr_train = s_rois
            train_dataset = RLCroppingDataset(
                sd_train, d_train, sr_train, r_train, m_train, patch_size,
                patch_size // 2,
            )

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(train_dataset, batch_size, True)

        # Validation
        if verbose > 1:
            print('< Validation dataset >')
        if source is None:
            val_dataset = LesionCroppingDataset(
                d_val, m_val, r_val, patch_size, 0, filtered=False,
            )
        else:
            val_dataset = DACroppingDataset(
                sd_val, d_val, sr_val, r_val, m_val, patch_size, 0
            )
        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(val_dataset, batch_size)
        if verbose > 1:
            print(
                'Training / validation samples = %d / %d' % (
                    len(train_dataset), len(val_dataset)
                )
            )

        if verbose > 0:
            n_param = sum(
                p.numel() for p in net.parameters()
                if p.requires_grad
            )
            print(
                '{:}Training with a {:} with residual blocks{:} '
                '({:}{:d}{:} parameters)'.format(
                    c['c'], c['g'] + model_type, c['nc'],
                    c['b'], n_param, c['nc'],
                )
            )
        net.fit(train_loader, val_loader, epochs=epochs, patience=patience)
        net.save_model(os.path.join(d_path, model_name))


def test(net, image, roi, path=None, unet_name=None, mask_name=None):
    bb = get_bb(roi)

    # Testing
    seg_im = np.zeros_like(roi)
    seg_bb = net.lesions(image)
    if len(seg_bb.shape) > 3:
        seg_im[bb] = np.argmax(seg_bb, axis=0) + 1
    else:
        seg_im[bb] = seg_bb > 0.5
    seg_im[np.logical_not(roi)] = 0

    if path is not None:
        # Reference data
        ref_name = find_file(mask_name, path)
        ref_nii = load_nii(ref_name)
        ref_mask = get_mask(ref_name)

        # Image saving
        out_name = os.path.join(path, unet_name)
        seg_nii = nib.Nifti1Image(seg_im, ref_nii.get_qform(), ref_nii.header)
        seg_nii.to_filename(out_name)

        pr_im = np.zeros_like(roi).astype(np.float32)
        if len(seg_bb.shape) > 3:
            for i, seg_bbi in enumerate(seg_bb):
                pr_im[bb] = seg_bbi
                pr_im[np.logical_not(roi)] = 0
                out_name = os.path.join(
                    path, 'parts.flair_pr{:}.nii.gz'.format(i)
                )
                seg_nii = nib.Nifti1Image(
                    pr_im, ref_nii.get_qform(), ref_nii.header
                )
                seg_nii.to_filename(out_name)

        return remove_small_regions(seg_im == 1), ref_mask

    else:
        return remove_small_regions(seg_im == 1)


def da(
        unet, lesion_dicts, source_dicts, t_path, model_name, gt_name,
        flair_name, mask_name, dataset, test_enabled=False, verbose=0
):
    c = color_codes()
    # source_dicts = prepare_msseg_flair(t_path, verbose)
    source_testing = [(p, t) for p in source_dicts for t in p['t']]
    if verbose > 1:
        print(
            '{:}[{:}] {:}Starting adaptation to {:} with {:}VISMS{:}'.format(
                c['c'], strftime("%H:%M:%S"), c['g'], dataset, c['y'], c['nc']
            )
        )
    da_unet = DomainAdapter(n_images=1)
    train(
        da_unet, model_name, lesion_dicts, val_split=0, source=source_testing,
        model_type='DAnet', verbose=verbose
    )
    for p_les, t_les in source_testing:
        if test_enabled:
            p_path = os.path.join(t_path, p_les['name'])
            bb = get_bb(t_les['roi'])
            t_im = t_les['images'][(slice(None),) + bb]
            if verbose > 1:
                print(
                    '{:}Testing patient (adapted) {:}'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc']
                    )
                )
            lesions_da, gt = test(
                da_unet, t_im, t_les['roi'], p_path,
                'da_unet.flair_mask.nii.gz', gt_name
            )

            if verbose > 1:
                print(
                    '{:}Testing patient (not adapted) {:}'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc']
                    )
                )

            # We convert the images to the source domain and store them.
            mask_im = load_nii(find_file(mask_name, p_path)).get_fdata()
            big_roi = mask_im > 0
            big_bb = get_bb(big_roi)
            big_im = t_les['images'][(slice(None),) + big_bb]
            big_mask = np.expand_dims(big_roi[big_bb], axis=0)
            fake_crop = da_unet.transform(big_im, big_mask)

            main_nii = load_nii(find_file(flair_name, p_path))
            main_im = main_nii.get_fdata()
            main_mean = np.mean(main_im[big_roi])
            main_std = np.std(main_im[big_roi])
            fake_im = np.zeros_like(main_im)
            fake_im[big_bb] = fake_crop * main_std + main_mean
            fake_im[np.logical_not(big_roi)] = 0
            flair_nii = nib.Nifti1Image(
                fake_im, main_nii.get_qform(), main_nii.header
            )
            flair_nii.to_filename(os.path.join(p_path, 'FLAIR_visms.nii.gz'))

            # We segment the lesions.
            lesions, gt = test(
                unet, t_im, t_les['roi'], p_path, 'og_unet.flair_mask.nii.gz',
                gt_name
            )

            d_da = patient_analysis(gt, lesions_da, p_les['name'])
            d = patient_analysis(gt, lesions, p_les['name'])
            if verbose > 0:
                print(
                    '{:}||{:^6}|{:^6}|{:^6}||{:^6}|{:^6}||{:^6}|{:^6}|'
                    '|{:^6}|{:^6}|'.format(
                        c['clr'] + p_les['name'],
                        'TPFV', 'FPFV', 'DSCV', 'TPFL', 'FPFL',
                        'TPL', 'GTL', 'Vox', 'GTV'
                    )
                )
                print(
                    '  DAnet  ||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                    '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                        d_da['tpfv'], d_da['fpfv'], d_da['dscv'],
                        d_da['tpfl'], d_da['fpfl'],
                        d_da['tp'], d_da['gt_d'],
                        d_da['lesion_s'], d_da['gt_s'],

                    )
                )
                print(
                    '  OGnet  ||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                    '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                        d['tpfv'], d['fpfv'], d['dscv'],
                        d['tpfl'], d['fpfl'],
                        d['tp'], d['gt_d'],
                        d['lesion_s'], d['gt_s'],

                    )
                )

    return da_unet


def train_visms(test_enabled=False, verbose=0):
    # Init
    c = color_codes()
    d_path = parse_inputs()['visms_dir']

    # Training (VISMS)
    lesion_dicts = prepare_visms_patients(d_path, verbose)
    model_name = 'unet-baseline.flair.visms.pt'
    unet = LesionsUNet(n_images=1)
    train(unet, model_name, lesion_dicts, val_split=0, verbose=verbose)
    if test_enabled:
        lesion_testing = [(p, t) for p in lesion_dicts for t in p['t']]
        if verbose > 0:
            print(
                ' Patient (timepoint) ||{:^6}|{:^6}|{:^6}||{:^6}|{:^6}|'
                '|{:^6}|{:^6}||{:^6}|{:^6}|'.format(
                    'TPFV', 'FPFV', 'DSCV', 'TPFL', 'FPFL',
                    'TPL', 'GTL', 'Vox', 'GTV'
                )
            )

        tpfv_list = []
        fpfv_list = []
        dscv_list = []
        tpfl_list = []
        fpfl_list = []
        tpl_list = []
        gtl_list = []
        vox_list = []
        gtv_list = []
        for p_les, t_les in lesion_testing:
            t_path = os.path.join(d_path, p_les['name'], t_les['name'])
            bb = get_bb(t_les['roi'])
            if verbose > 1:
                print(
                    '{:}Testing patient {:} (t: {:})'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc'],
                                c['c'] + t_les['name'] + c['nc']
                    ),
                    end='\r'
                )
            t_im = t_les['images'][(slice(None),) + bb]
            lesions, gt = test(
                unet, t_im, t_les['roi'], t_path,
                'unet.flair_mask.nii.gz',
                'lesion_mask.nii.gz'
            )
            d = patient_analysis(gt, lesions, p_les['name'])
            if verbose > 0:
                print(
                    '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                    '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                        c['clr'],
                        '{:} (t: {:})'.format(d['name'], t_les['name']),
                        d['tpfv'], d['fpfv'], d['dscv'],
                        d['tpfl'], d['fpfl'],
                        d['tp'], d['gt_d'],
                        d['lesion_s'], d['gt_s'],

                    )
                )
            tpfv_list.append(d['tpfv'])
            fpfv_list.append(d['fpfv'])
            dscv_list.append(d['dscv'])
            tpfl_list.append(d['tpfl'])
            fpfl_list.append(d['fpfl'])
            tpl_list.append(d['tp'])
            gtl_list.append(d['gt_d'])
            vox_list.append(d['lesion_s'])
            gtv_list.append(d['gt_s'])
        if verbose > 0:
            print(
                '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                    c['clr'], 'Average',
                    np.mean(tpfv_list), np.mean(fpfv_list), np.mean(dscv_list),
                    np.mean(tpfl_list), np.mean(fpfl_list),
                    np.sum(tpl_list), np.sum(gtl_list),
                    np.sum(vox_list), np.sum(gtv_list)

                )
            )

    # Domain adaptation (MSSEG2016)
    # lesion_dicts = prepare_visms_flair(d_path, verbose)
    model_name = 'unet-msseg.da.flair.visms.pt'
    path = parse_inputs()['msseg_dir']
    source_dicts = prepare_msseg_patients(path, verbose)
    da_unet = da(
        unet, lesion_dicts, source_dicts, path, model_name,
        'Consensus.nii.gz', 'FLAIR_preprocessed.nii.gz',
        'T1_preprocessed.nii.gz', 'MSSEG', test_enabled, verbose
    )

    if test_enabled:
        lesion_testing = [(p, t) for p in lesion_dicts for t in p['t']]
        if verbose > 0:
            print(
                ' Patient (timepoint) ||{:^6}|{:^6}|{:^6}||{:^6}|{:^6}|'
                '|{:^6}|{:^6}||{:^6}|{:^6}|'.format(
                    'TPFV', 'FPFV', 'DSCV', 'TPFL', 'FPFL',
                    'TPL', 'GTL', 'Vox', 'GTV'
                )
            )

        tpfv_list = []
        fpfv_list = []
        dscv_list = []
        tpfl_list = []
        fpfl_list = []
        tpl_list = []
        gtl_list = []
        vox_list = []
        gtv_list = []
        for p_les, t_les in lesion_testing:
            t_path = os.path.join(d_path, p_les['name'], t_les['name'])
            bb = get_bb(t_les['roi'])
            if verbose > 1:
                print(
                    '{:}Testing patient {:} (t: {:})'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc'],
                                c['c'] + t_les['name'] + c['nc']
                    ),
                    end='\r'
                )
            t_im = t_les['images'][(slice(None),) + bb]
            lesions, gt = test(
                da_unet, t_im, t_les['roi'], t_path,
                'unet.msseg.flair_mask.nii.gz',
                'lesion_mask.nii.gz'
            )
            d = patient_analysis(gt, lesions, p_les['name'])
            if verbose > 0:
                print(
                    '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                    '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                        c['clr'],
                        '{:} (t: {:})'.format(d['name'], t_les['name']),
                        d['tpfv'], d['fpfv'], d['dscv'],
                        d['tpfl'], d['fpfl'],
                        d['tp'], d['gt_d'],
                        d['lesion_s'], d['gt_s'],

                    )
                )
            tpfv_list.append(d['tpfv'])
            fpfv_list.append(d['fpfv'])
            dscv_list.append(d['dscv'])
            tpfl_list.append(d['tpfl'])
            fpfl_list.append(d['fpfl'])
            tpl_list.append(d['tp'])
            gtl_list.append(d['gt_d'])
            vox_list.append(d['lesion_s'])
            gtv_list.append(d['gt_s'])
        if verbose > 0:
            print(
                '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                    c['clr'], 'Average',
                    np.mean(tpfv_list), np.mean(fpfv_list), np.mean(dscv_list),
                    np.mean(tpfl_list), np.mean(fpfl_list),
                    np.sum(tpl_list), np.sum(gtl_list),
                    np.sum(vox_list), np.sum(gtv_list)

                )
            )


    model_name = 'unet-lit.da.flair.visms.pt'
    path = parse_inputs()['lit_dir']
    source_dicts = prepare_lit_patients(path, verbose)
    da_unet = da(
        unet, lesion_dicts, source_dicts, path, model_name,
        'consensus_gt.nii.gz', 'flair_corrected.nii.gz',
        'brainmask.nii.gz', 'LIT', test_enabled, verbose
    )

    if test_enabled:
        lesion_testing = [(p, t) for p in lesion_dicts for t in p['t']]
        if verbose > 0:
            print(
                ' Patient (timepoint) ||{:^6}|{:^6}|{:^6}||{:^6}|{:^6}|'
                '|{:^6}|{:^6}||{:^6}|{:^6}|'.format(
                    'TPFV', 'FPFV', 'DSCV', 'TPFL', 'FPFL',
                    'TPL', 'GTL', 'Vox', 'GTV'
                )
            )

        tpfv_list = []
        fpfv_list = []
        dscv_list = []
        tpfl_list = []
        fpfl_list = []
        tpl_list = []
        gtl_list = []
        vox_list = []
        gtv_list = []
        for p_les, t_les in lesion_testing:
            t_path = os.path.join(d_path, p_les['name'], t_les['name'])
            bb = get_bb(t_les['roi'])
            if verbose > 1:
                print(
                    '{:}Testing patient {:} (t: {:})'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc'],
                                c['c'] + t_les['name'] + c['nc']
                    ),
                    end='\r'
                )
            t_im = t_les['images'][(slice(None),) + bb]
            lesions, gt = test(
                da_unet, t_im, t_les['roi'], t_path,
                'unet.lit.flair_mask.nii.gz',
                'lesion_mask.nii.gz'
            )
            d = patient_analysis(gt, lesions, p_les['name'])
            if verbose > 0:
                print(
                    '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                    '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                        c['clr'],
                        '{:} (t: {:})'.format(d['name'], t_les['name']),
                        d['tpfv'], d['fpfv'], d['dscv'],
                        d['tpfl'], d['fpfl'],
                        d['tp'], d['gt_d'],
                        d['lesion_s'], d['gt_s'],

                    )
                )
            tpfv_list.append(d['tpfv'])
            fpfv_list.append(d['fpfv'])
            dscv_list.append(d['dscv'])
            tpfl_list.append(d['tpfl'])
            fpfl_list.append(d['fpfl'])
            tpl_list.append(d['tp'])
            gtl_list.append(d['gt_d'])
            vox_list.append(d['lesion_s'])
            gtv_list.append(d['gt_s'])
        if verbose > 0:
            print(
                '{:}{:<21}||{:6.3f}|{:6.3f}|{:6.4f}||{:6.3f}|{:6.3f}|'
                '|{:6d}|{:6d}||{:6d}|{:6d}|'.format(
                    c['clr'], 'Average',
                    np.mean(tpfv_list), np.mean(fpfv_list), np.mean(dscv_list),
                    np.mean(tpfl_list), np.mean(fpfl_list),
                    np.sum(tpl_list), np.sum(gtl_list),
                    np.sum(vox_list), np.sum(gtv_list)

                )
            )


def baseline(
        d_path, lesion_dicts, gt_name='Consensus', dataset='MSSEG',
        test_enabled=False, verbose=0
):
    # Init
    c = color_codes()

    # Training
    print(
        '{:}Preparing {:} {:}baseline{:} model (no DA){:}'.format(
            c['c'], c['y'] + dataset, c['g'], c['nc'] + c['c'], c['nc']
        )
    )

    # Leave-one-out approach (only 15 cases)
    for i, p_les in enumerate(lesion_dicts):
        t_les = p_les['t'][0]
        train_dicts = lesion_dicts[:i] + lesion_dicts[i + 1:]
        model_name = 'unet-{:}_baseline.flair.{:}.pt'.format(
            dataset.lower(), p_les['name']
        )
        unet = LesionsUNet(n_images=1)
        if verbose > 1:
            print(
                '{:}Training patient (lesion) {:} (t: {:})'.format(
                    c['c'], c['g'] + p_les['name'] + c['nc'],
                            c['c'] + t_les['name'] + c['nc']
                )
            )
        train(
            unet, model_name, train_dicts, verbose=verbose
        )

        if test_enabled:
            p_path = os.path.join(d_path, p_les['name'])
            bb = get_bb(t_les['roi'])
            if verbose > 1:
                print(
                    '{:}Testing patient (lesion) {:}'.format(
                        c['c'], c['g'] + p_les['name'] + c['nc'],
                    )
                )
            t_im = t_les['images'][(slice(None),) + bb]
            test(
                unet, t_im, t_les['roi'], p_path,
                'baseline_unet.flair_mask.nii.gz', gt_name
            )


def eval_msseg_nets(verbose=0):
    # Init
    c = color_codes()
    d_path = parse_inputs()['visms_dir']
    lesion_dicts = prepare_msseg_patients(parse_inputs()['msseg_dir'], verbose)

    print('{:}Evaluating the models (MSSEG){:}'.format(c['g'], c['nc']))
    print(
        '{:}{:^15}|{:^6}|{:^6}|{:^6}|{:^6}|'
        '|{:^6}|{:^6}|{:^6}|{:^6}|'.format(
            c['clr'], 'Netname', 'TPFL', 'FPFL', 'DSCV', 'DSCG',
            'P r2', 'log r2', 'S ρ', 'K τ',
        )
    )
    lesion_testing = [(p, t) for p in lesion_dicts for t in p['t']]
    net = LesionsUNet(n_images=1)

    # Leave-one-out approach with MSSEG data
    model_name = 'unet-msseg_baseline.flair.{:}.pt'
    general_analysis(
        net, lesion_testing, 'bl.flair.msseg', model_name, verbose=verbose
    )

    # VISMS pretrained model
    model_name = 'unet-baseline.flair.visms.pt'
    net.load_model(os.path.join(d_path, model_name))
    general_analysis(net, lesion_testing, 'na.flair.msseg', verbose=verbose)

    # VISMS pretrained model + adaptation
    model_name = 'unet-msseg.da.flair.visms.pt'
    net = DomainAdapter(n_images=1)
    general_analysis(
        net, lesion_testing, 'da.flair.msseg', model_name, verbose=verbose
    )


def eval_nets(lesion_dicts, dataset='MSSEG', verbose=0):
    # Init
    c = color_codes()
    d_path = parse_inputs()['visms_dir']

    print(
        '{:}Evaluating the models ({:}){:}'.format(
            c['g'], c['y'] + dataset + c['g'], c['nc']
        )
    )
    print(
        '{:}{:^15}|{:^6}|{:^6}|{:^6}|{:^6}|'
        '|{:^6}|{:^6}|{:^6}|{:^6}|'.format(
            c['clr'], 'Netname', 'TPFL', 'FPFL', 'DSCV', 'DSCG',
            'P r2', 'log r2', 'S ρ', 'K τ',
        )
    )
    lesion_testing = [(p, t) for p in lesion_dicts for t in p['t']]
    net = LesionsUNet(n_images=1)

    # Leave-one-out approach with the dataset
    model_name = 'unet-' + dataset.lower() + '_baseline.flair.{:}.pt'
    general_analysis(
        net, lesion_testing, 'bl.flair.{:}'.format(dataset.lower()),
        model_name, verbose=verbose
    )

    # VISMS pretrained model
    model_name = 'unet-baseline.flair.visms.pt'
    net.load_model(os.path.join(d_path, model_name))
    general_analysis(
        net, lesion_testing, 'na.flair.{:}'.format(dataset.lower()),
        verbose=verbose
    )

    # VISMS pretrained model + adaptation
    model_name = 'unet-' + dataset.lower() + '.da.flair.visms.pt'
    net = DomainAdapter(n_images=1)
    general_analysis(
        net, lesion_testing, 'da.flair.{:}'.format(dataset.lower()),
        model_name, verbose=verbose
    )


"""
> Dummy main function
"""


def main():
    # Init
    c = color_codes()

    print(
        '{:}[{:}] {:}<Lesion segmentation>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' < VISMS (with and without adapatation > '''
    print(
        '{:}[{:}] {:}Starting overfit to {:}VISMS{:} (FLAIR only)'.format(
            c['c'], strftime("%H:%M:%S"), c['g'], c['y'], c['nc']
        )
    )
    train_visms(test_enabled=parse_inputs()['run_test'], verbose=2)

    print(
        '{:}[{:}] {:}Starting baseline{:} (FLAIR only)'.format(
            c['c'], strftime("%H:%M:%S"), c['g'], c['nc']
        )
    )
    options = parse_inputs()
    lesion_dicts = prepare_msseg_patients(options['msseg_dir'], 2)
    baseline(
        options['msseg_dir'], lesion_dicts, dataset='MSSEG',
        gt_name='Consensus.nii.gz',
        test_enabled=options['run_test'], verbose=2
    )
    lesion_dicts = prepare_lit_patients(options['lit_dir'], 2)
    baseline(
        options['lit_dir'], lesion_dicts, dataset='LIT',
        gt_name='consensus_gt.nii.gz',
        test_enabled=options['run_test'], verbose=2
    )

    print(
        '{:}[{:}] {:}Starting evaluation{:} (FLAIR only)'.format(
            c['c'], strftime("%H:%M:%S"), c['g'], c['nc']
        )
    )
    lesion_dicts = prepare_msseg_patients(options['msseg_dir'], verbose=2)
    eval_nets(lesion_dicts, dataset='MSSEG', verbose=2)
    lesion_dicts = prepare_msseg_patients(options['lit_dir'], verbose=2)
    eval_nets(lesion_dicts, dataset='LIT', verbose=2)


if __name__ == '__main__':
    main()
