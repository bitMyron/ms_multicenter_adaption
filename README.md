
# Introduction
This model performs cross-sectional lesion identification.

# Environment
At least one NVIDIA GPU with >=4GB memory + NVIDIA Runtime.

# How to build into a Docker container
```console
cd /path/to/repo/LesionCrossSec
docker build -t lesion-cross-sec:1.0 .
```

# How to run as a container
1. Put your data in case folders under */path/to/data/folder*. For example:
*/path/to/data/folder/case_01*
*/path/to/data/folder/case_02*
*/path/to/data/folder/patient23*
...

	The name of the case folder is used as the identifier of each case.

2. Each case folder needs to have the following files present:
	- *flair_brain_mni.nii.gz*  -> Flair image in MNI space and with pixels outside the brain masked out
	- *t1_brain_mni.nii.gz*     -> T1 image in MNI space and with pixels outside the brain masked out
	- *flair_brain.nii.gz*               -> Flair image in its original raw space and with pixels outside the brain masked out
	- *flair2mni.mat*              -> FLAIR space to MNI space affine transformation matrix (obtained from *flirt*)

3. Run docker container:
	```console
    docker run -it --rm -v /path/to/data/folder:/input -v /path/to/output:/output lesion-cross-sec:1.0
   ```
4. Output is stored under each of the input case folder created under the folder mapped to */output* as:
	- *lesion_mask_unet3d.full_mni.nii.gz* -> Lesion mask produced by the model that was trained by a 70-30 train-validation split
	- *lesion_mask_unet3d_mni.nii.gz* -> Lesion mask produced by the model that was trained by a 100-0 train-validation split

# Contact
For any question please contact Andy Shieh from SNAC andy.shieh@snac.com.au