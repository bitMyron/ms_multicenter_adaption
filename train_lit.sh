set -e # Return error code when things fail
CUDA_VISIBLE_DEVICES=2,3

# The example folder should contain the cases in the following format:
# input_folder/case/flair_brain_mni.nii.gz  -> MNI space
# input_folder/case/t1_brain_mni.nii.gz     -> MNI space
# input_folder/case/flair_brain.nii.gz      -> Original space
# input_folder/case/flair2mni.mat           -> FLAIR to MNI affine transformation
#
# The names of the files can be different, but that basically means changing part of the code.
# The results for each patient will be stored as:
# output_folder/case/lesion_mask_unet3d_mni.nii.gz -> MNI space
# output_folder/case/lesion_mask_unet3d.nii.gz     -> Original FLAIR space
#
# I put both original FLAIR and MNI space just in case, but if moving back to original is not
# needed, it's as easy as commenting the function convert_to_original in lines 711 - 714 (main.py).
#
# The weights are defined on the function test_folder (line 654 main.py) and take into account the images.
# By default it's FLAIR+T1.
[ -e nohup.out ] && rm nohup.out
nohup python main.py -d /home/mayang/data/LIT -o /home/mayang/data/output/LIT -t lit --run-train --dropout 0.25 --epochs 20 --patience 5 --filters 32/64/128/256/512 &

