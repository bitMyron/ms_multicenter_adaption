"""
The main file running inside the docker (the starting point)
"""
# Import the required packages
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [2,3])
import argparse
from pytorch.train_unet import test_folder, train_full_model, cross_train_test

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
        '--run_cross_train',
        dest='run_cross_train',
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


def main():
    """
    Dummy main function.
    """

    args = parse_args()
    # Training with all cases
    if args['run_train']:
        train_full_model(args, verbose=1)
    if args['run_test']:
        test_folder(args, verbose=1)
        # test_folder(
        #     net_name='lesions.full-unet.{:}_model.pt', suffix='unet3d.full',
        #     verbose=1
        # )
    if args['run_cross_train']:
        cross_train_test(args, verbose=1)


if __name__ == "__main__":
    main()
