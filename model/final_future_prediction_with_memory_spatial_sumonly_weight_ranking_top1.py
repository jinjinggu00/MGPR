import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *
from hf2_memory import *

import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from .Attention import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4,  tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()
        self.memory_hf2vad = Memory_hf2vad(num_slots=1000, slot_dim=1382400,
                                           shrink_thres=0.0005) if True else None
        self.conv1x1 = torch.nn.Conv2d(256, 128,kernel_size=1)
        self.channel_att = channel_attention(channels=1024)
        self.space_att = Attention()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)


        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(64, n_channel, 32)

    def forward(self, x,  skip2, skip3):
        x_att = self.channel_att(x)
        print('asfdsfa', x.shape)
        tensorConv = self.moduleConv(x_att)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        tensorUpsample4 = self.space_att(tensorUpsample4)
        # tensorUpsample4 = self.conv1x1(tensorUpsample4)   # 4x128x90x160

        bs, C, H, W = tensorUpsample4.shape
        tensorUpsample4 = tensorUpsample4.reshape(bs, -1)
        memory_hf2vad_out = self.memory_hf2vad(tensorUpsample4)
        tensorUpsample4 = memory_hf2vad_out["out"]
        att_weigth1 = memory_hf2vad_out["att_weight"]
        tensorUpsample4 = tensorUpsample4.reshape(bs, C, H, W)   # 4x128x90x160
        # print('the shape of the output for moduleUpsample4 is :', tensorUpsample4.shape)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)  # 沿着通道
        # print('the size after cat is :', cat4.shape)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        print(tensorUpsample2.shape)
        # cat2 = torch.cat((skip1, tensorUpsample2), dim=1)
        # print(cat2.shape)

        output = self.moduleDeconv1(tensorUpsample2)

        return output, att_weigth1


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1,
                 temp_gather=0.1):
        super(convAE, self).__init__()


        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):

        fea,  skip2, skip3 = self.encoder(x)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output, att_weight = self.decoder(updated_fea,  skip2, skip3)

            return output, att_weight, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss

        # test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.memory(
                fea, keys, train)
            output, att_weight = self.decoder(updated_fea,  skip2, skip3)
            return output,att_weight, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss







