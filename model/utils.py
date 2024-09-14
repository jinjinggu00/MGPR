import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data

import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, flows_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.flows_dir = flows_folder
        self.transform = transform
        self.videos = OrderedDict()
        self.flows = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))   # ['./dataset/ped2/testing/frames/12', './dataset/ped2/testing/frames/04', './dataset/ped2/testing/frames/06',

        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

        flows = glob.glob(os.path.join(self.flows_dir, '*'))
        for flow in sorted(flows):
            video_name = flow.split('/')[-1]
            self.flows[video_name] = {}
            self.flows[video_name]['path'] = flow
            self.flows[video_name]['flow'] = glob.glob(os.path.join(flow, '*.jpg'))
            self.flows[video_name]['flow'].sort()
            self.flows[video_name]['length'] = len(self.flows[video_name]['flow'])

    def get_all_samples(self):
        frames = []
        flows = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        flow_videos = glob.glob(os.path.join(self.flows_dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        for flow_video in sorted(flow_videos):
            video_name = flow_video.split('/')[-1]
            for i in range(len(self.flows[video_name]['flow']) - self._time_step):
                flows.append(self.flows[video_name]['flow'][i])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])


        batch = []
        batch_flows = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width)
            flow = np_load_frame(self.flows[video_name]['flow'][frame_name + i], self._resize_height,
                                 self._resize_width)

            if self.transform is not None:
                batch.append(self.transform(image))
                batch_flows.append(self.transform(flow))

        return np.concatenate(batch, axis=0), np.concatenate(batch_flows, axis=0)

    def __len__(self):
        return len(self.samples)
