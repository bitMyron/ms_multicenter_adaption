# Packages and Global Variables
import os
import torch
import shutil
import warnings
import numpy as np
from glob import glob
import nibabel as nib
from chaoyi.MSBaseNet import MSBaseNet
from torch.autograd import Variable
from collections import OrderedDict
model_path = 'best_model.pkl'
testing_dataset = './sasha_dataset_testing/{}/{}.nii.gz'
output_path = './output_sample/'
# Function Declaration
def renew_folder(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict
def prepare_data_loading():
    dataset_paths = []
    flair_paths = glob(testing_dataset.format('flair', '*'))
    for flair_path in flair_paths:
        t1_path = flair_path.replace('flair', 't1')
        path_dict = {'flair':flair_path, 't1':t1_path}
        dataset_paths.append(path_dict)
    return dataset_paths
def make_square_padding(im):
    height, width = im.shape[0], im.shape[1]
    size = 256
    top_pad = int((size - height) / 2)
    left_pad = int((size - width) / 2)
    new_im = np.zeros((size, size, 3))
    new_im[top_pad: top_pad+height, left_pad:left_pad + width] = im
    return new_im, [top_pad, top_pad+height, left_pad, left_pad+width]
def img_preprocess(tensor, loader_mean):
    tensor = tensor.astype(np.float64)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor * 255
    tensor -= loader_mean
    tensor = 2 * tensor
    tensor = tensor.astype(float) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    return tensor
def test():
    dataset_mean = np.array([122.67892, 116.66877, 104.00699])
    renew_folder(output_path)
    print('Model (path) to be loaded:', model_path)
    print('Dataset to be loaded: sasha-testing')
    print('Output path: {}'.format(output_path))
    # Setup Model
    model = MSBaseNet()
    state = convert_state_dict(torch.load(model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    # Setup image
    print('Start Inference -------->')
    dataset_paths = prepare_data_loading()
    for paths_dict in dataset_paths:
        t1_path, flair_path = paths_dict['t1'], paths_dict['flair']
        t1_mod = nib.load(t1_path).get_data()
        flair = nib.load(flair_path)
        flair_mod = flair.get_data()
        output_lesion_map = np.zeros(flair_mod.shape)
        print("Read input from : {}, input_flair.shape: {}".format(t1_path.split('/')[-1].split('.')[0], flair_mod.shape))
        for z_index in range(1,t1_mod.shape[2]-1):
            t1_slice = np.array(make_square_padding(t1_mod[:, :, z_index - 1:z_index + 2])[0])
            t1_slice = img_preprocess(t1_slice, dataset_mean)
            flair_slice, [a, b, c, d] = make_square_padding(flair_mod[:, :, z_index - 1:z_index + 2])
            flair_slice = img_preprocess(np.array(flair_slice)[:, :, ::-1], dataset_mean)
            slice_input = np.concatenate((t1_slice, flair_slice), axis=0)
            slice_input = np.expand_dims(slice_input, 0)
            slice_input = torch.from_numpy(slice_input).float()
            if torch.cuda.is_available():
                model.cuda(0)
                with torch.no_grad():
                    images = Variable(slice_input.cuda(0))
            else:
                with torch.no_grad():
                    images = Variable(slice_input)
            output = torch.argmax(model(images), dim = 1).data.cpu().numpy()[0]
            output_lesion_map[:, :, z_index] = output[a:b, c:d]
        output_tensor = nib.Nifti1Image(output_lesion_map, flair.affine)
        save_path = output_path + '{}.nii.gz'.format(t1_path.split('/')[-1].split('.')[0])
        nib.save(output_tensor, save_path)
        print("     Save output to : {}, output.shape: {}".format(save_path, output_lesion_map.shape))
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    test()





