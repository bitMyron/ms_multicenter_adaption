import os

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [2, 3])
import argparse
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
    get_data, get_isbi_data, get_lit_data, get_messg_data, get_case, cross_validation_split,
    cross_validation_split_isbi,
    get_case_seperate_addr
)
from tools.lesion_manipulation import (
    remove_small_regions
)
from tools.lesion_metrics import get_lesion_metrics
from data_manipulation.utils import get_bb


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
        '-d', '--datafile-path',
        dest='datafile_path',
        default='/home/yangma/1/07003SATH/FLAIR_preprocessed.nii.gz;/home/yangma/1/07003SATH/T1_preprocessed.nii.gz',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-b', '--brainmask-path',
        dest='brainmask_path',
        default='/home/yangma/1/07003SATH/Mask_registered.nii.gz',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-l', '--lesionmask-path',
        dest='lesionmask_path',
        default='/home/yangma/1/07003SATH/Consensus.nii.gz',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-o', '--output-path',
        dest='output_path',
        default='/home/yangma/1/output',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-m', '--model-path',
        dest='model_path',
        default='/home/yangma/1/lesions-unet.flair.t1_model.pt',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-f', '--filters',
        dest='filters',
        default='32;64;128;256',
        help='Parameter to store the working directory.'
    )
    parser.add_argument(
        '-g', '--gpu',
        dest='gpu_id',
        type=int, default=0,
        help='GPU id number.'
    )
    parser.add_argument(
        '--metric_file',
        dest='metric_file',
        type=str, default='metrics.csv',
        help='GPU id number.'
    )
    parser.add_argument(
        '--general-flag',
        dest='general_flag',
        action='store_true', default=True,
        help='Whether to test a network on the working directory.'
    )
    return vars(parser.parse_args())

def test_net(
        args, save_pr=True, brain_name=None, verbose=1, task=None
):
    """
    Function that tests a fully trained network with a set of testing images.
    :param args: arguments from the main funciton.
    :param save_pr: Whether to save the probability maps or not.
    :param brain_name: Name of the brain mask image.
    :param verbose: Verbosity level
    :return: Lists of the most important metrics and a dictionary with all
     the computed metrics per patient.
    """
    # Init
    datafile_path = args['datafile_path'].split(';')
    brainmask_path = args['brainmask_path']
    lesionmask_path = args['lesionmask_path']
    filters = [int(fi) for fi in args['filters'].split(';')]
    model_path = args['model_path']
    o_path = args['output_path']
    metric_file = open(os.path.join(o_path, args['metric_file']), 'w')
    general_flag = args['general_flag']

    # Load the already trained network
    net = LesionsUNet(
        conv_filters=filters, n_images=len(datafile_path), dropout=0
    )
    net.load_model(model_path)

    # Load the test object's data
    testing, tst_brain, gt_lesion_mask, spacing, example_nii = get_case_seperate_addr(
        d_file_path=datafile_path, lm_file_path=lesionmask_path, bm_file_path=brainmask_path
    )

    # try:
    #     seg_bb = net.lesions(
    #         testing, verbose=verbose
    #     )
    # except RuntimeError:
    seg_bb = net.patch_lesions(
        testing, patch_size=64,
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

    # Saving the predicted lesion mask with the same nii header as the ground truth
    mask_nii = nib.Nifti1Image(
        lesion_unet,
        example_nii.get_qform(),
        example_nii.get_header()
    )
    mask_nii.to_filename(
        os.path.join(
            o_path, 'lesion_mask.nii.gz'
        )
    )

    # If ground truth mask exists, calculate eval metrics
    if gt_lesion_mask is not None:
        test_case_dsc = get_lesion_metrics(gt_lesion_mask, lesion_unet, spacing, metric_file=metric_file, patient='infer', general_flag=general_flag)
        print("%s\n" % str(test_case_dsc))
    metric_file.close()


def main():
    """
    Dummy main function.
    """
    args = parse_args()
    test_net(args)

if __name__ == "__main__":
    main()
