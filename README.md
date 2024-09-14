# MRPR
Official pytorch implementation of Unsupervised Video Anomaly Detection forMultiple Traffic Scenes via Memory-Guided FramePrediction and Reconstruction

# Environment
Following here to prepare the environment of FlowNet2.0: https://github.com/NVIDIA/flownet2-pytorch

pytorch = 1.10.0

python = 3.8.16

tqdm = 4.65.0

torchvision = 0.11.0

# Preprocess
1.Extracting the optial flows by:

                python ./preprocess/flows.py

2.Convert the optical flow files in numpy format to RGB images by(You may need to rectify the dataset path in numpy_to_flow_img.py):

                python ./preprocess/numpy_to_flow_img.py

# Train
Run our proposed method by:

                python .main.py

# Test
1.We provide pre-trained models. You can download from here:https://pan.baidu.com/s/1Q7jFT2E7yk_uxrMQgxqTfg?pwd=njc1 .code:njc1.

2.Place the weight file according to the path specified in back_bone.py.

3.Run:

    python ./eval/eva_2.py
