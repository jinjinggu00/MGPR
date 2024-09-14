import os
import numpy as np
import cv2
import shutil

def convert_optical_flow_to_images(input_folder, output_folder):
    # 遍历输入文件夹下的所有子目录及其包含的所有numpy文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                # 读取numpy文件
                file_path = os.path.join(root, file)
                flow = np.load(file_path)

                # 将光流数据转换为图像
                hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 获取输出子文件夹路径（保持与输入文件夹相对路径一致）
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                # 创建输出子文件夹（如果不存在）
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # 生成输出文件路径
                filename = os.path.splitext(os.path.basename(file_path))[0]
                name = filename.split('.')[0]
                output_file_path = os.path.join(output_subfolder, f"{name}.jpg")

                # 保存光流图像
                cv2.imwrite(output_file_path, bgr)

# 指定输入文件夹和输出文件夹路径
input_folder = '../dataset/gu/training/flows'
output_folder = '../dataset/gu/training/flows_image'

# 删除输出文件夹（如果存在），以便重新生成
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# 调用函数进行转换
convert_optical_flow_to_images(input_folder, output_folder)

