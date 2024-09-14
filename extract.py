import cv2
import os

def extract_frames(video_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 初始化帧计数器
    frame_count = 0

    while True:
        # 读取视频的一帧
        ret, frame = video.read()

        # 如果没有读取到帧，则退出循环
        if not ret:
            break

        # 构造保存帧的文件名
        frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count}.jpg"

        # 保存帧图像到文件
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # 打印进度信息
        print(f"Extracted frame {frame_count}")

        # 增加帧计数器
        frame_count += 1

    # 释放视频对象
    video.release()


# 调用函数进行批量抽帧
video_folder = "/home/zhangheng/Desktop/12.7data/"
output_folder = "/home/zhangheng/Desktop/12.7data/test/"

for root, dirs, files in os.walk(video_folder):
    for file in files:
        video_path = os.path.join(root, file)
        extract_frames(video_path, output_folder)