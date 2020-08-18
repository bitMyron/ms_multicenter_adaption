import _pickle as cp
import numpy as np
class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.grouped = False
        self.preds = {}
        self.gts = {}

    def update(self, label_trues, label_preds, location_infos):
        assert label_preds.shape[0] == label_trues.shape[0] == len(location_infos), 'Dimension NOT MATCH'
        for fileidx, info in enumerate(location_infos):
            tokens = info.split('-')
            caseidx = tokens[0]
            sliceidx = int(tokens[1])
            if caseidx not in self.preds.keys():
                self.preds[caseidx] = {}
                self.gts[caseidx] = {}
            if sliceidx not in self.preds[caseidx].keys():
                self.preds[caseidx][sliceidx] = label_preds[fileidx, :, :]
                self.gts[caseidx][sliceidx] = label_trues[fileidx, :, :]
    def group_slices(self):
        if self.grouped:
            return
        self.grouped = True
        for caseidx in self.preds.keys():
            preds3d, gts3d = [], []
            caseidxlist = list(self.preds[caseidx].keys())
            start_idx = min(caseidxlist)
            end_idx = max(caseidxlist) + 1
            for sliceidx in range(start_idx, end_idx):
                preds3d.append(self.preds[caseidx][sliceidx])
                gts3d.append(self.gts[caseidx][sliceidx])
            self.preds[caseidx] = np.array(preds3d)
            self.gts[caseidx] = np.array(gts3d)

    def cal_dice(self, a, b):
        a, b = np.array(a), np.array(b)
        common = np.sum(np.logical_and(a, b))
        dice = common * 2 / (np.sum(a) + np.sum(b))
        return dice
    def get_scores(self):
        # group by case
        self.group_slices()

        all_preds, all_gts = [], []
        subject_dice_map = {}
        for caseidx in self.preds.keys():
            pred, gt = self.preds[caseidx], self.gts[caseidx]
            # subject - dice
            subject_dice_map[caseidx] = self.cal_dice(pred, gt)
            # voxel - dice
            all_preds.extend(pred.flatten())
            all_gts.extend(gt.flatten())
        voxel_dice = self.cal_dice(all_preds, all_gts)
        return {'Subject-Dice: \t': subject_dice_map,
                'Voxel-Dice: \t': voxel_dice,
                }

    def reset(self):
        self.preds = {}
        self.gts = {}
        self.grouped = False
