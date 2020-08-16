from operator import add
from functools import partial
import os
import sys
from nibabel import load as load_nii
import numpy as np
from .utils import print_message, find_file, slicing
import sitk


def atlas_registration(
        atlas,
        reference,
        atlases_pr,
        mask,
        structures=None,
        path=None,
        patient='',
        timepoint='',
        patch_size=(3, 3, 3),
        affine_sampling=0.75,
        affine_steps=100,
        affine_levels=3,
        demons_steps=200,
        demons_sigma=0.5,
        verbose=1,

):
    """
    Function to perform atlas registration using Demons from SimpleITK.
    :param atlas: Path to the template image file for the atlas.
    :param reference: Path to the refence image file for the atlas.
    :param atlases_pr: List of paths to the probabilistic atlas files.
    :param mask: Path to the brainmask image.
    :param structures: Path to a structural mask for the atlas.
    :param path: Path where the output will be saved.
    :param patient: Name of the patient being processed.
    :param timepoint: Timepoint of the patient.
    :param patch_size: Patch size used for the similarity image.
    :param affine_sampling: Sampling for affine registration.
    :param affine_steps: Steps for the affine registration.
    :param affine_levels: Levels of the Demons registration.
    :param demons_steps: Steps for the Demons registration.
    :param demons_sigma: Sigma for the Demons registration.
    :param verbose: Verbose levels for this tool. The minimum value must be 1.
     For this level of verbosity, only "required" messages involving each step
     and likelihood will be shown. For the next level, various debugging
     options related to the expectation maximisation will be shown.
    """
    # Init
    patch_half = tuple(map(lambda ps: ps / 2, patch_size))

    # > Atlas registration
    # Affine registration for Demons.
    # if not find_file('atlas_affine.nii.gz', temp_path):
    if verbose > 0:
        print_message(
            'Atlas registration - %s (%s)' % (timepoint, patient)
        )
    if verbose > 1:
        print(
            '- Affine registration - %s (%s)' % (timepoint, patient)
        )
    affine = sitk.itkaffine(
        reference, atlas, sampling=affine_sampling,
        steps=affine_steps, levels=affine_levels
    )
    sitk.itkresample(
        reference, atlas, affine,
        path=path, name='atlas_affine'
    )
    if structures is not None:
        sitk.itkresample(
            reference, structures, affine, interpolation='nn',
            path=path, name='structures_affine'
        )
    [
        sitk.itkresample(
            reference, pr_i, affine,
            path=path, name='atlas_affine_pr%d' % i
        ) for pr_i, i in zip(atlases_pr, range(len(atlases_pr)))
    ]

    # Histogram matching
    atlas_affine = os.path.join(path, 'atlas_affine.nii.gz')
    if verbose > 1:
        print(
            '- Histogram matching - %s (%s)' % (timepoint, patient)
        )
    sitk.itkhist_match(
        reference, atlas_affine, match_points=24,
        path=path, name='atlas'
    )

    # Demons computation
    if verbose > 1:
        print(
            '- Demons registration - %s (%s)' % (timepoint, patient)
        )
    atlas_matched = os.path.join(
        path, 'atlas_corrected_matched.nii.gz'
    )
    sitk.itkdemons(
        reference, atlas_matched, mask, path=path, name='atlas',
        steps=demons_steps, sigma=demons_sigma
    )

    df = os.path.join(path, 'atlas_multidemons_deformation.nii.gz')
    if structures is not None:
        sitk.itkwarp(
            reference, os.path.join(path, 'structures_affine.nii.gz'),
            df, interpolation='nn', path=path, name='atlas_ventricles'
        )
    [
        sitk.itkwarp(
            reference, pr_i, df,
            path=path, name='atlas_pr%d' % i
        ) for i, pr_i in enumerate(
            map(
                lambda i: os.path.join(path, 'atlas_affine_pr%d.nii.gz' % i),
                range(len(atlases_pr))
            )
        )
    ]
    sitk.itkwarp(
        reference, atlas_matched, df,
        path=path, name='atlas_demons'
    )

    # > Similarity computing
    # First we pad the images to extract the mask patches
    if verbose > 0:
        print_message(
            'Atlas similarity - %s (%s)' % (timepoint, patient)
        )

    if find_file('atlas_similarity_xcor.nii.gz', path) is None:
        t1nii = load_nii(reference)
        t1 = t1nii.get_data()
        atlas_demons = load_nii(
            os.path.join(path, 'atlas_demons.nii.gz')
        ).get_data()
        padding = tuple(
            (idx, size - idx) for idx, size in zip(patch_half, patch_size)
        )
        f_pad = np.pad(t1, padding, 'constant', constant_values=0.0)
        m_pad = np.pad(atlas_demons, padding, 'constant', constant_values=0.0)

        # Then we compute the centers (slices) for the mask voxels
        mask_im = load_nii(mask).get_data()
        centers = [
            tuple(idx) for idx in np.stack(np.nonzero(mask_im), axis=1)
        ]
        [x, y, z] = np.stack(centers, axis=1)
        new_centers = map(
            lambda center: map(add, center, patch_half),
            centers
        )
        [slices_x, slices_y, slices_z] = slicing(new_centers, patch_size)

        # Finally, we just compute cross-correlation on the patches and
        # save the information inside the voxel
        similarity = np.zeros_like(t1)
        f_pad_s = np.stack(
            np.split(f_pad[slices_x, slices_y, slices_z], len(centers)),
            axis=1
        )
        m_pad_s = np.stack(
            np.split(m_pad[slices_x, slices_y, slices_z], len(centers)),
            axis=1
        )
        # The variable is called cross variance, because it would be the
        # variance if the patches were the same.
        f_var = f_pad_s - f_pad_s.mean(axis=0)
        m_var = m_pad_s - m_pad_s.mean(axis=0)
        slices_xvar = np.mean(f_var * m_var, axis=0)
        slices_std = f_pad_s.std(axis=0) * m_pad_s.std(axis=0)
        slices_std[slices_std <= 0] = np.finfo(float).eps
        slices_xcorr = np.fabs(slices_xvar / slices_std)
        slices_xcorr[slices_xcorr > 1] = 1
        similarity[x, y, z] = slices_xcorr
        t1nii.get_data()[:] = similarity
        t1nii.to_filename(os.path.join(path, 'atlas_similarity_xcor.nii.gz'))

        if verbose > 1:
            print(
                '- Similarity range = [{:f}, {:f}]'.format(
                    similarity.min(initial=None), similarity.max(initial=None)
                )
            )

    if find_file('atlas_similarity.nii.gz', path) is None:
        t1nii = load_nii(reference)
        t1 = t1nii.get_data()
        mask_im = load_nii(mask).get_data()
        atlas_demons = load_nii(
            os.path.join(path, 'atlas_demons.nii.gz')
        ).get_data()

        # Everything on the official similarity will be based on subtraction.
        sub = np.abs(t1 - atlas_demons) * mask_im
        similarity = 1 - sub / sub.max()

        # Finally, we just compute cross-correlation on the patches and
        # save the information inside the voxel
        t1nii.get_data()[:] = similarity
        t1nii.to_filename(os.path.join(path, 'atlas_similarity.nii.gz'))

        if verbose > 1:
            print(
                '- Similarity range = [%f, %f]' % (
                    similarity.min(), similarity.max()
                )
            )


def expectation(data, probability, threshold=0.0, verbose=1):
    # Data should have shape (n_images, x_size, y_size, z_size)
    thresholded_pr = probability > threshold
    voxel_int = np.stack(map(lambda v: v[thresholded_pr], data))
    voxel_pr = np.tile(probability[thresholded_pr], (len(data), 1))

    # Mean computation using only the voxels with a probability higher than threshold
    mu = (voxel_int * voxel_pr).mean(axis=1)

    # Sigma computation
    sigma = np.cov(voxel_int, aweights=probability[thresholded_pr])

    if verbose > 1:
        print('--- mu = [%s]' % ', '.join(map(lambda x: '%.5f' % x, mu)))
        print('--- sigma diagonal = [%s]' % ', '.join(map(lambda x: '%.5f' % x, np.diag(sigma))))

    return mu, sigma


def maximisation(data, roi, mu, sigma, verbose=1):
    # Data should have shape (n_images, x_size, y_size, z_size)
    d = len(data)
    prob = np.zeros(data.shape[1:])
    roi_bool = roi.astype(dtype=np.bool)
    data_roi = np.stack(map(lambda im: im[roi_bool], data))
    sigma_det = np.linalg.det(sigma) if np.linalg.det(sigma) != 0 else np.finfo(np.float32).eps
    left = 1 / (np.power(2 * np.pi, d) * sigma_det)
    data_mu = (np.expand_dims(mu, axis=1) - data_roi).T
    try:
        right = np.sum(
            -0.5 * np.matmul(data_mu, np.linalg.pinv(sigma)) * data_mu, axis=1
        )
    except np.linalg.LinAlgError:
        right = np.sum(
            -0.5 * np.matmul(data_mu, np.linalg.inv(sigma)) * data_mu, axis=1
        )
    if verbose > 1:
        print(
            '--- right_range [%.2e, %.2e] / left value (cov det) = %.2e (%.2e)' % (
                np.min(right), np.max(right), left, sigma_det
            )
        )

    prob[roi_bool] = left * np.exp(right)

    return prob


def tissue_pve(
        image_names,
        mask_name,
        atlas_names,
        path=None,
        patient='',
        timepoint='',
        th=0.75,
        max_iter=10,
        alpha=2.0,
        pv_classes=None,
        mixed_classes=None,
        verbose=1,
):
    """
    Function that performs atlas registration and tissue segmentation using a
     probabilistic atlas that differentiates between cortical CSF and the
     ventricles.
    :param image_names: List of paths to the images to be used for segmentation.
    :param mask_name: Path to the mask of the brain.
    :param atlas_names: Probabilistic atlases.
    :param path: Path where the output will be saved.
    :param patient: Name of the patient being processed.
    :param timepoint: Timepoint of the patient.
    :param th: Threshold for the tissue segmentation step. This threshold is
     used for the trimmed likelihood estimator during the expectation
     maximisation approach. Unlike previous C++ versions of this method, this
     threshold is adaptive to allow the mean and covariance computation even
     with values under this threshold for each class.
    :param max_iter: Max number of iterations for the expectation maximisation
     approach.
    :param alpha: Parameter for the threshold estimation during the "lesion
     segmentation" thresholding.
    :param pv_classes: List of tuple pairs of values that represent the tissue
     classes. For example, 0: Ventricles-CSF, 1: External-CSF, 2: GM, 3: WM.
    :param mixed_classes: List of the subclasses that pertain to the same
     tissue class. For example, if we assume that the ventricals and external
     CSF are classes 0 and 1, we would have a mixed class (0, 1). Non-mixed
     classes are defined by the index of the prior atlas. For example, if we
     consider the example defined in pv_classes, the mixed_classes would be:
     [(0, 1), 2, 3] -> (0, 1) for CSF, 2 for GM and 3 for WM.
    :param verbose: Verbose levels for this tool. The minimum value must be 1.
     For this level of verbosity, only "required" messages involving each step
     and likelihood will be shown. For the next level, various debugging
     options related to the expectation maximisation will be shown.
    :return: None.
    """

    # Init
    if verbose > 0:
        print_message(
            'Tissue segmentation - %s (%s)' % (timepoint, patient)
        )
    if pv_classes is None:
        pv_classes = []
    if mixed_classes is None:
        mixed_classes = range(len(atlas_names))

    prnii = load_nii(atlas_names[0])
    atlases_pr = [
        load_nii(name).get_data().astype(np.float32) for name in atlas_names
    ]
    images = np.stack(
        map(
            lambda name: load_nii(name).get_data().astype(np.float32),
            image_names
        )
    )
    masknii = load_nii(mask_name)
    mask = masknii.get_data()
    flair = load_nii(image_names[-1]).get_data()

    '''Tissue segmentation'''
    if verbose > 1:
        print('- Segmentation start - %s (%s)' % (timepoint, patient))
    if find_file('brain.nii.gz', path) is None:
        # Init
        for a in atlases_pr:
            a[a < 0] = 0
        pure_tissues = len(atlases_pr)

        # Partial volume atlas creation and atlas probability normalisation
        if verbose > 1:
            print('- Partial volume class atlas creation')

        # Now we'll create the partial volume atlases. That means that we need
        # to merge the atlases of both classes, and renormalize everything.
        # Remember: The sum of all atlases for a given voxel should be 1.
        atlases = atlases_pr + [
            (atlases_pr[i0] + atlases_pr[i1]) / 2.0 for i0, i1 in pv_classes
        ]
        if verbose > 1:
            iapr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    atlases
                )
            )
            print('-- initial atlas ranges = %s)' % iapr_s)

        # Here we renormalize the probabilities.
        atlases_sum = np.sum(atlases, axis=0)
        if verbose > 1:
            print(
                '-- atlas sum ranges = [%.5f, %.5f]' % (
                    atlases_sum.min(initial=None), atlases_sum.max(initial=None)
                )
            )
        nonzero_sum = np.nonzero(atlases_sum)
        for a in atlases:
            a[nonzero_sum] = a[nonzero_sum] / atlases_sum[nonzero_sum]
        if verbose > 1:
            apr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    atlases
                )
            )
            print('-- atlas ranges = %s)' % apr_s)

        # First, we create the atlas priors. These are constant and are
        # computed using the atlas priors and the similarity image. Since
        # they are constant, we'll compute them once only. In the C++ code
        # these maps were recomputed at each iteration. We'll just do it once
        # here.
        # Pure tissue classes
        if verbose > 1:
            print('- Atlas priors (initial)')
        apr = map(np.copy, atlases)

        # Finally, we create the initial posterior probabilities. This are defined
        # by the Gauss distribution probability function. For pure tissues we
        # estimate the mu and sigma from the data and the atlas priors. For the
        # partial volumes we average them.
        # However, for the initial estimate, we'll use the priors and we'll
        # update cpr during the next iterations. For convenience I'm keeping
        # expectation and maximisation as functions for the current function.
        # They are before the loop for better readability.
        # Pure tissue classes
        if verbose > 1:
            print('- Posterior probabilities (initial)')
        ppr = map(np.copy, atlases)
        if verbose > 1:
            ppr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    ppr
                )
            )
            print('--  (ppr ranges = %s)' % ppr_s)

        # Initial values for loop
        sum_log_ant = -np.inf
        best_log = sum_log_ant
        best_ppr = map(np.copy, ppr)

        sum_ppr = np.sum(ppr, axis=0)
        sum_log = np.sum(map(
            lambda pr_i: np.sum(np.log(pr_i[pr_i > 0] / sum_ppr[pr_i > 0])),
            ppr
        ))
        i = 0
        if verbose > 0:
            print_message(
                '- Main EM loop (initial log likelihood = %.2e)' % sum_log
            )
        eps = np.finfo(float).eps
        while i < max_iter and np.fabs(sum_log_ant - sum_log) > eps:
            i += 1
            if verbose > 1:
                print('-- Iteration %2d' % i)
            elif verbose > 0:
                print('-- Iteration %2d' % i, end=' ')
                sys.stdout.flush()
            # <Expectation step>
            # The pure parameters are updated from the data, while the partial
            # ones are updated using the pure ones.
            min_pure_ppr = np.min(map(np.max, ppr[:pure_tissues]))
            adaptive_th = min_pure_ppr / 2.0 if min_pure_ppr < th else th
            if verbose > 1:
                print('--- expectation')
            elif verbose > 0:
                print('<expectation>', end=' ')
                sys.stdout.flush()

            pure_pr = map(
                lambda classes: np.sum(
                    map(
                        lambda c: ppr[c],
                        classes
                    ),
                    axis=0
                ) if isinstance(classes, tuple) or isinstance(classes, list)
                else ppr[classes],
                mixed_classes
            )

            pure_unique_params = [
                expectation(
                    images, pr_i, adaptive_th, verbose
                ) for pr_i in pure_pr
            ]

            pure_params = [None] * pure_tissues
            for k, classes in enumerate(mixed_classes):
                if isinstance(classes, tuple) or isinstance(classes, list):
                    for c in classes:
                        pure_params[c] = pure_unique_params[k]
                else:
                    pure_params[classes] = pure_unique_params[k]

            pv_params = [
                tuple(
                    [
                        (p0 + p1) / 2 for p0, p1 in
                        zip(pure_params[i0], pure_params[i1])
                    ]
                ) for i0, i1 in pv_classes
            ]
            params = pure_params + pv_params

            # <Maximisation step>
            # The conditional probability is computed using the Gaussian
            # mixture model defined previously. Since the mean and covariance
            # matrix are already updated, the conditional probability is
            # computed using the same equation for both pure and partial
            # classes.

            # Conditional probability (Gaussian)
            if verbose > 1:
                print('--- maximisation')
            elif verbose > 0:
                print('<maximisation>', end=' ')
                sys.stdout.flush()
            partial_max = partial(maximisation, data=images, roi=mask, verbose=verbose)
            cpr = [
                partial_max(mu_i, sigma_i) for mu_i, sigma_i in params
            ]
            # Priors: Atlas weighted by similarity + Neighbourhood weighted by
            # inverse similarity
            # Posterior probability = cpr * priors
            ppr = [mask * cpr_i * prior_i for cpr_i, prior_i in zip(cpr, apr)]

            # Posterior are normalised with the sum of the probabilities for
            # each class
            sum_ppr = np.sum(ppr, axis=0)
            nonzero_pr = np.nonzero(sum_ppr > 0)
            for ppr_i in ppr:
                ppr_i[nonzero_pr] = ppr_i[nonzero_pr] / sum_ppr[nonzero_pr]
                ppr_i[ppr_i < 0] = 0
                ppr_i[ppr_i > 1] = 1

            if verbose > 1:
                apr_s = ' '.join(
                    map(
                        lambda pr_i: '[{:.5f}, {:.5f}]'.format(pr_i.min(), pr_i.max()),
                        apr
                    )
                )
                print('--  (apr ranges = {:})'.format(apr_s))
                cpr_s = ' '.join(
                    [
                        '[{:.2e}, {:.2e}]'.format(
                            pr_i.min(), pr_i.max()
                        ) for pr_i in cpr
                    ]
                )
                print(
                    '--  (conditional probability ranges = {:})'.format(cpr_s)
                )
                ppr_s = ' '.join(
                    map(
                        lambda pr_i: '[{:f}, {:f}]'.format(pr_i.min(), pr_i.max()),
                        ppr
                    )
                )
                print('-- (posterior probability ranges = %s)' % ppr_s)

            # Update the objective function
            sum_log_ant = sum_log
            sum_log = np.sum(
                map(lambda pr_i: np.sum(np.log(pr_i[pr_i > 0])), ppr)
            )

            if sum_log > best_log:
                best_log = sum_log
                best_ppr = map(np.copy, ppr)

            if verbose > 1:
                print('-- Log-likelihood = %.2e' % sum_log)
            elif verbose > 0:
                print('log-likelihood = %.2e' % sum_log)

        # We save the probability maps
        for i, pr in enumerate(best_ppr):
            prnii.get_data()[:] = pr
            prnii.to_filename(os.path.join(path, 'tissue_pr%d.nii.gz' % i))

        brain = np.squeeze(
            np.argmax(ppr, axis=0) + 1
        ).astype(mask.dtype) * mask

        # We'll find lesions by thresholding.
        if verbose > 0:
            print_message('- Lesion segmentation')
        flair_roi = flair[brain == 2]
        if verbose > 1:
            print('-- Threshold estimation')
        mu = flair_roi.mean()
        sigma = flair_roi.std()

        t = mu + alpha * sigma
        if verbose > 1:
            print('-- Threshold: %f (%f + %f * %f)' % (t, mu, alpha, sigma))

        wml = (flair * mask) > t
        brain[wml] = brain.max() + 1

        masknii.get_data()[:] = brain
        masknii.to_filename(os.path.join(path, 'brain.nii.gz'))
