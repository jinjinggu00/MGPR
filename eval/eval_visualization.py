import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from utils import DataLoader
# from models.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from models.Reconstruction import *
from sklearn.metrics import roc_auc_score
from eval_utils import *
import random
import glob
import xlrd
import argparse
from back_bone import Backbone


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=6, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=240, help='height of input images')
parser.add_argument('--w', type=int, default=360, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=256, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=256, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, default='../free_way/ped2_model_60ep.pth', help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='../free_way/ped2_keys_60ep.pt', help='directory of model')

args = parser.parse_args()





METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    # "avenue": {
    #    "testing_video_num": 21,
    #    "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
    #                           472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
    #                           76]
    # },

    "ped2(gao su lu)": {
        "testing_video_num": 15,
        "testing_frames_cnt": [92, 56, 84, 47, 60, 40, 30, 105, 124, 55, 41, 63, 46, 33, 50]
    },

    # "shanghaitech": {
    #     "testing_video_num": 107,
    #     "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
    #                            337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
    #                            337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
    #                            313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
    #                            385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
    #                            505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
    #                            409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
    #                            409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
    #                            457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
    #                            361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
    #                            241, 289, 361, 385, 217, 337, 265]
    # },

    "shanghaitech": {
        "testing_video_num": 14,
        "testing_frames_cnt": [572, 293, 537, 169, 119, 560, 120, 90, 458, 138,
                               505, 218, 314, 182]
    },

    "hzroad": {
        "testing_video_num": 1,
        "testing_frames_cnt": [127]
    },

}




device = 'cuda'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"
flows_folder = args.dataset_path + "/" + args.dataset_type + "/testing/flows_image"

# Loading dataset
test_dataset = DataLoader(test_folder, flows_folder, transforms.Compose([
    transforms.ToTensor(),
]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

from final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *

# Loading the trained model
#model = torch.load(args.model_dir)
#model.cuda()
# m_items = torch.load(args.m_items_dir)
#m_items = torch.load('./check/flows_keys.pt')
if args.dataset_type=='ped2':
    labels = np.load('../frame_labels_ped2.npy')
    print('labels',labels.shape, len(labels), labels )

if args.dataset_type=='freeway':
    data = xlrd.open_workbook("../free_way_test.xls")
    sheet = data.sheet_by_name('Sheet1')

    gt = {}
    for ii in range(sheet.ncols):
        # gt[ii] = np.asarray(sheet.row_values(ii))
        gt[ii] = list(sheet.col_values(ii))
        gt[ii] = [jj for jj in gt[ii] if jj != '']
        # print('len the tables',len(gt[ii]))

    content = np.concatenate(list(gt.values()), axis=0)
    labels= content.reshape(1, 935)
    # print(labels)
# print('len the labels', labels.size)



videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}
ssim_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])

    else:
        labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length'] + label_length])
        print(len(labels_list))
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

    ssim_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']

#m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).to(device)
#m_items_test = m_items.clone()

model = Backbone()
device = 'cuda'



model.eval()
from SSIM import *

with torch.no_grad():
    for k, (imgs, flows) in enumerate(test_batch):

        if args.method == 'pred':
            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
        else:
            if k == label_length:
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        imgs = Variable(imgs).cuda()
        flows = Variable(flows).cuda()



        outputs, att_weight, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss, flow_out = model.forward(
            imgs[:, 0:3 * 4], flows[:, 0:3 * 4],  False)
        print(outputs.shape, k)
        mse_imgs = torch.mean(loss_func_mse((outputs[0] ) , (imgs[0, 3 * 4:]  ) )).item()
        mse_flows = torch.mean(loss_func_mse((flow_out[0]) , (flows[0, 3 * 4:]  ) ))
        mse_feas = compactness_loss.item()
        # print('zhang zhang ', outputs.shape, imgs[0:,3*4:].shape )
        # ssim_val = ssim_loss(outputs, imgs[0:, 12:])

        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
        ssim_list[videos_list[video_num].split('/')[-1]].append(mse_flows)



# Measuring the abnormality score and the AUC


anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    # print(len(anomaly_score_list(psnr_list[video_name])) ,psnr_list[video_name])
    ssim_list_in = tensor_to_list(ssim_list[video_name])
    # print(len(anomaly_score_list(psnr_list[video_name])),anomaly_score_list(psnr_list[video_name]))
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
                                     anomaly_score_list_inv(feature_distance_list[video_name]),ssim_list_in, args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)
all_frame_score = anomaly_score_total_list

print(type(all_frame_score), all_frame_score.shape,all_frame_score)


# Measuring the abnormality score and the AUC


new_gt = np.array([])
new_frame_scores = np.array([])


dataset_name = 'ped2'
start_idx = 0
for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
    gt_each_video = labels_list[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
    scores_each_video = all_frame_score[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]

    start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

    print('biao zhu ', new_gt.shape, gt_each_video.shape, gt_each_video)

    new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
    new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

gt_concat = new_gt
all_frame_score = new_frame_scores

# accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))




from visual_utils import *

suffix="best"
curves_save_path = os.path.join('./', 'freeway_test_result', 'anomaly_curves_%s' % suffix)
auc = save_evaluation_curves(all_frame_score, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

print('The result of ', args.dataset_type)
print('AUC: ', auc * 100, '%')
