import time
import math
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from torch.utils import data
from chaoyi.MSBaseNet import MSBaseNet
from chaoyi.CY_metrics import runningScore
import torch.backends.cudnn as cudnn
from chaoyi.CYDataset_MSLesion import CYDataset_MSLesion
from chaoyi.lr_scheduling import poly_lr_scheduler, resnet_lr_scheduler


def cy_utils_create_directory(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def prepare_class_weights(ratio: object, device: object) -> object:
    weight = torch.ones(1)
    weight = (weight * ratio).to(device)
    return weight

def train():
    DATASET_ROOT = './data/SNAC_Lesion_ID_Proj_Prepro_TrainTestSplit'
    IMG_SIZE = 256
    NUM_EPOCHES = 300
    ratio = 8
    BATCH_SIZE = 16 * ratio
    LR = 0.01 * math.sqrt(ratio)
    LR_REDUCE_STEP = 15
    GPU_INDEX = -1  # -1 to use all GPU
    NUM_WORKERS = 16
    CE_WEIGHT = 10
    PrintBatchInterval = 100
    EvalEpochInterval = 5
    EXP_ID = 'EXP_' + str(random.randint(0, 10000))
    print('ExperimentID={}'.format(EXP_ID))
    result_folder = './ExpResults/{}/'.format(EXP_ID)
    cy_utils_create_directory(result_folder)
    print('IMG_SIZE={}'.format(IMG_SIZE))
    print('NUM_EPOCHES={}'.format(NUM_EPOCHES))
    print('BATCH_SIZE={}'.format(BATCH_SIZE))
    print('LR={}'.format(LR))
    print('GPU_INDEX={}'.format(GPU_INDEX))
    print('NUM_WORKERS={}'.format(NUM_WORKERS))
    print('PrintBatchInterval={}'.format(PrintBatchInterval))
    print('EvalEpochInterval={}'.format(EvalEpochInterval))
    print('BCEwithLogitsLoss-POS_WEIGHT={}'.format(CE_WEIGHT))
    print('#################################################################')
    trainset = CYDataset_MSLesion(DATASET_ROOT, 'train', IMG_SIZE)
    validationset = CYDataset_MSLesion(DATASET_ROOT, 'test', IMG_SIZE)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    validationloader = data.DataLoader(validationset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Setup Model
    model = MSBaseNet()

    if GPU_INDEX == -1:
        print('GPUs available: {}'.format(torch.cuda.device_count()))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        pos_weight = prepare_class_weights(CE_WEIGHT, device)
    else:
        print('GPU[{}] is being used.'.format(GPU_INDEX))
        device = torch.device("cuda:{}".format(GPU_INDEX) if torch.cuda.is_available() else "cpu")
        model.to(device)
        pos_weight = prepare_class_weights(CE_WEIGHT, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Setup Metrics
    running_metrics = runningScore(trainset.num_classes)

    print('#################################################################')
    best_dice = -100.0
    start = datetime.datetime.now()
    for epoch in range(NUM_EPOCHES):
        model.train()
        loss_train_sum = 0
        start_time = time.time()
        if (epoch + 1) % LR_REDUCE_STEP == 0:
            LR = LR / 10
            print('decrease lr by 10, to [{}]'.format(LR))
            resnet_lr_scheduler(optimizer, LR)
        for batch_counter, (images, labels, location_infos) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # Start - learning rate varing
            iter = len(trainloader) * epoch + batch_counter
            poly_lr_scheduler(optimizer, LR, iter)
            # End - learning rate varing
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs[:, 1, :, :], labels)
            loss.backward()
            optimizer.step()

            loss_train_sum += loss.item()
            
            if batch_counter % PrintBatchInterval == 0:
                now = datetime.datetime.now()
                duration = now - start
                print('---{} | Batch [{:05d}/{:05d}] | Duration {:.1f}s | loss: {:.7f}'.format(now.strftime("%c"), batch_counter, len(trainloader), duration.total_seconds(), loss.item()))
                start = datetime.datetime.now()
        avg_train_loss = loss_train_sum / len(trainloader)
        print('Epoch[{}]-Summary-Training: Loss={:.7f} | Time={:.4f}'.format(epoch+1,
                                                                               avg_train_loss,
                                                                               time.time() - start_time))
        print('-----------------------------------------------------')
        if epoch % EvalEpochInterval == 0:
            model.eval()
            loss_test_sum = 0
            for _, (images_val, labels_val, location_infos) in enumerate(validationloader):
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)

                optimizer.zero_grad()
                outputs_val = model(images_val)
                loss = criterion(outputs_val[:, 1, :, :], labels_val)

                # running_metrics
                _, preds_map_bin_val = outputs_val.max(1)
                running_metrics.update(labels_val.data.cpu().numpy(),
                                       preds_map_bin_val.cpu().numpy(),
                                       location_infos)

                loss_test_sum += loss.item()
            avg_test_loss = loss_test_sum / len(validationloader)
            print('Epoch[{}]-Summary-Testing: Loss={:.7f}'.format(epoch+1, avg_test_loss))
            score = running_metrics.get_scores()
            curr_voxel_dice = score['Voxel-Dice: \t']
            for k, v in score.items():
                if k == 'Subject-Dice: \t':
                    print(k, )
                    subject_dice_list = []
                    for caseidx, subject_dice in v.items():
                        subject_dice_list.append(subject_dice)
                        print('\t\t', caseidx + ':\t', subject_dice)
                    avg_subject_dice = np.mean(np.array(subject_dice_list))
                    std_subject_dice = np.std(np.array(subject_dice_list))
                    print('\t AVG:', avg_subject_dice, '+-', std_subject_dice)
                else:
                    print(k, v)
            running_metrics.reset()

            if curr_voxel_dice >= best_dice:
                best_dice = curr_voxel_dice
                state = {'epoch': epoch + 1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), }
                save_path = "best_model.pkl"
                torch.save(state,
                           result_folder + save_path)
                print('[BEST_MODEL] model saved at {}.'.format(save_path))
            print('#################################################################')
if __name__ == '__main__':
    train()
