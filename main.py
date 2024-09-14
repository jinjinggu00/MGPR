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
from models.utils import DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm
import argparse

import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

from SSIM import *


parser = argparse.ArgumentParser(description="MNAD")
#parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=6, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=360, help='height of input images')
parser.add_argument('--w', type=int, default=640, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

args = parser.parse_args()
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# torch.backends.cudnn.enabled = False




# model seting
assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from models.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    print('build the model success!!')
else:
    from models.Reconstruction import *
    model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)

model.to(device)


input = torch.rand(4, 12, 360, 640).to(device)
m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).to(device)

outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(input, m_items, True)

print(outputs.shape)
'''



torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

train_size = len(train_dataset)

test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

print(len(train_batch))
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert args.method == 'pred' or args.method == 'recon', 'Wrong task name'
if args.method == 'pred':
    from models.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim).cuda()

    print('build the model success !')
else:
    from models.Reconstruction import *
    model = convAE(args.c, memory_size = args.msize, feature_dim = args.fdim, key_dim = args.mdim)
    print('build the model success !')

params_encoder =  list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
#model.cuda()
#model.to(device)


if torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')
    device_id = [0, 1]
    model = nn.DataParallel(model, device_id=device_id)



loss_func_mse = nn.MSELoss(reduction='none')


x = torch.randn(5, 12, 240, 360).to(device)

m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).to(device)

# torch.onnx.export(model, (x, m_items, True),'./net.pth')

#outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(x, m_items, True)

for epoch in range(args.epochs):
    labels_list = []
    model.train()

    start = time.time()

    for idx, imgs in tqdm(enumerate(train_batch), total=len(train_batch)):
        # imgs = Variable(imgs).cuda()
        imgs = imgs.to(device)
        print(imgs.shape)


        if args.method == 'pred':
            outputs, att_weight, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:,0:12], m_items, True)
            #print(outputs.shape)
        else:
            outputs, att_weight, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)

        optimizer.zero_grad()
        mem3_loss = torch.mean(torch.sum(att_weight * torch.log(att_weight + 1e-12), dim=1))
        ssim_val = ssim_loss(outputs, imgs[:, 12:])
        if args.method == 'pred':
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))

        else:
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs))

        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss + mem3_loss + ssim_val
        loss.backward(retain_graph=True)
        optimizer.step()

    scheduler.step()

    print('----------------------------------------')
    print('Epoch:', epoch + 1)
    if args.method == 'pred':
        print('Loss: Prediction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}/ SSIM_Loss {:.6f}'.format(loss_pixel.item(),
                                                                                        compactness_loss.item(),
                                                                                        separateness_loss.item(),
                                                                                        ssim_val.item()))
    else:
        print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(),
                                                                                            compactness_loss.item(),
                                                                                            separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')

print('Training is finished')
# Save the model and the memory items


log_dir = './exp/free_way'
torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f
sys.stdout = orig_stdout
f.close()


