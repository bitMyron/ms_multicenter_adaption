from glob import glob
import nibabel as nib
import numpy as np
pred_base_path = './output_sample/{}.nii.gz'
gt_paths = glob('sasha_dataset_testing/gt/*.nii.gz')
def cal_dice(a, b):
	a, b = np.array(a), np.array(b)
	common = np.sum(np.logical_and(a, b))
	dice = common * 2 / (np.sum(a) + np.sum(b))
	return dice
subject_dices = []
all_voxels_gt, all_voxels_pred = [], []
for gt_path in gt_paths:
	# loading	
	scan_idx = gt_path.split('/')[-1].split('.')[0]
	
	pred_path = pred_base_path.format(scan_idx)
	pred = nib.load(pred_path).get_data()
	gt = nib.load(gt_path).get_data()
	# subject dice
	subject_dice = cal_dice(gt, pred)
	subject_dices.append(subject_dice)
	print(scan_idx, ': subject_dice={}'.format(subject_dice))
 	# voxel dice
	all_voxels_gt.extend(gt.flatten())
	all_voxels_pred.extend(pred.flatten())
print('Average-Subject-Dice:{}'.format(np.mean(np.array(subject_dices))))
print('Voxel-Dice:{}'.format(cal_dice(all_voxels_gt, all_voxels_pred)))
	
