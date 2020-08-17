from data_manipulation.metrics import (
    average_surface_distance, tp_fraction_seg, fp_fraction_seg, dsc_seg,
    tp_fraction_det, fp_fraction_det, dsc_det, true_positive_det, num_regions,
    num_voxels, probabilistic_dsc_seg, analysis_by_sizes
)

def get_lesion_metrics(gt_lesion_mask, lesion_unet, spacing, metric_file, patient, general_flag=True, fold=1):
    if general_flag:
        dist = average_surface_distance(gt_lesion_mask, lesion_unet, spacing)
        tpfv = tp_fraction_seg(gt_lesion_mask, lesion_unet)
        fpfv = fp_fraction_seg(gt_lesion_mask, lesion_unet)
        dscv = dsc_seg(gt_lesion_mask, lesion_unet)
        tpfl = tp_fraction_det(gt_lesion_mask, lesion_unet)
        fpfl = fp_fraction_det(gt_lesion_mask, lesion_unet)
        dscl = dsc_det(gt_lesion_mask, lesion_unet)
        tp = true_positive_det(lesion_unet, gt_lesion_mask)
        gt_d = num_regions(gt_lesion_mask)
        lesion_s = num_voxels(lesion_unet)
        gt_s = num_voxels(gt_lesion_mask)
        pdsc = probabilistic_dsc_seg(gt_lesion_mask, lesion_unet)
        if metric_file:
            metric_file.write(
                '%s;%s;%s;%f;%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n' % (
                    patient + 'gt', patient + 'pd', str(fold),
                    dist, tpfv, fpfv, dscv,
                    tpfl, fpfl, dscl,
                    tp, gt_d, lesion_s, gt_s
                )
            )
        else:
            print(
                'SurfDist TPFV FPFV DSCV '
                'TPFL FPFL DSCL '
                'TPL GTL Voxels GTV PrDSC'
            )
            print(
                '%f %f %f %f %f %f %f %d %d %d %d %f' % (
                    dist, tpfv, fpfv, dscv,
                    tpfl, fpfl, dscl,
                    tp, gt_d, lesion_s, gt_s, pdsc
                )
            )
    else:
        sizes = [3, 11, 51]
        tpf, fpf, dscd, dscs = analysis_by_sizes(gt_lesion_mask, lesion_unet, sizes)
        names = '%s;%s;' % (patient + 'gt', patient + 'pd')
        measures = ';'.join(
            [
                '%f;%f;%f;%f' % (tpf_i, fpf_i, dscd_i, dscs_i)
                for tpf_i, fpf_i, dscd_i, dscs_i in zip(
                tpf, fpf, dscd, dscs
            )
            ]
        )
        if metric_file:
            metric_file.write(names + measures + '\n')
        else:
            intervals = [
                '\t\t[%d-%d)\t\t|' % (mins, maxs)
                for mins, maxs in zip(sizes[:-1], sizes[1:])
            ]
            intervals = ''.join(intervals) + \
                        '\t\t[%d-inf)\t|' % sizes[-1]
            measures_s = 'TPF\tFPF\tDSCd\tDSCs\t|' * len(sizes)
            measures = ''.join(
                [
                    '%.2f\t%.2f\t%.2f\t%.2f\t|' % (
                        tpf_i, fpf_i, dscd_i, dscs_i
                    )
                    for tpf_i, fpf_i, dscd_i, dscs_i in zip(
                    tpf, fpf, dscd, dscs
                )
                ]
            )
            print(intervals)
            print(measures_s)
            print(measures)
