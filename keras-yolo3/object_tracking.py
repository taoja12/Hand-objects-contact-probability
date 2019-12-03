# -*- coding: utf-8 -*-
# from tracker import Tracker
import colorsys
import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from KalmanFilterTracker import Tracker  # 加载卡尔曼滤波函数
from yolo import YOLO
from scipy import misc


def calc_center(out_boxes, out_classes, out_scores, score_limit=0.5):
    outboxes_filter = []
    for x, y, z in zip(out_boxes, out_classes, out_scores):
        if z > score_limit:
            if y == 0:
                outboxes_filter.append(x)

    centers = []
    number = len(outboxes_filter)
    tracknum = []
    for box in outboxes_filter:
        top, left, bottom, right = box
        center = np.array([[(left + right) // 2], [(top + bottom) // 2]])
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)
        temppoint = (center_x, center_y)
        tracknum.append(temppoint)
        centers.append(center)
    return centers, number, tracknum


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    colors = [(0, 0, 255) if c == (255, 0, 0) else c for c in colors]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


def trackerDetection(tracker, image, centers, number, max_point_distance=30, max_colors=10, track_id_size=0.8):
    """
        - max_point_distance为两个点之间的欧式距离不能超过30
            - 有多条轨迹,tracker.tracks;
            - 每条轨迹有多个点,tracker.tracks[i].trace
        - max_colors,最大颜色数量
        - track_id_size,每个
    """
    # track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #            (0, 255, 255), (255, 0, 255), (255, 127, 255),
    #            (127, 0, 255), (127, 0, 127)]
    track_colors = get_colors_for_classes(max_colors)

    result = np.asarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(result, str(number), (20, 40), font, 1, (0, 0, 255), 5)  # 左上角，人数计数

    if (len(centers) > 0):
        # Track object using Kalman Filter
        tracker.Update(centers)
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            # 多个轨迹
            if (len(tracker.tracks[i].trace) > 1):
                x0, y0 = tracker.tracks[i].trace[-1][0][0], tracker.tracks[i].trace[-1][1][0]
                # cv2.putText(result, str(tracker.tracks[i].track_id), (int(x0), int(y0)), font, track_id_size,
                #             (255, 255, 255), 3)
                # (image,text,(x,y),font,size,color,粗细)
                for j in range(len(tracker.tracks[i].trace) - 1):
                    # 每条轨迹的每个点
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j + 1][0][0]
                    y2 = tracker.tracks[i].trace[j + 1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    # if distance < max_point_distance:
                    #     cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
                    #              track_colors[clr], 3)
    return tracker, result


if __name__ == '__main__':
    # 加载keras yolov3 voc预训练模型
    yolo_test_args = {
        "model_path": 'model_data/yolo_20000.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    yolo_test = YOLO(**yolo_test_args)

    pic_path = "/home/tao/projects/test_img/zheng_test/"
    pic_list = os.listdir(pic_path)
    num = len(pic_list)
    tracker = Tracker(100, 8, 15, 100)
    for n in tqdm(range(num)):
        image = Image.open(pic_path + '%s.jpg' % (n + 1))
        r_image, out_boxes, out_scores, out_classes = yolo_test.detect_image(image)
        centers, number, tracknum = calc_center(out_boxes, out_classes, out_scores, score_limit=0.5)
        tracker, result = trackerDetection(tracker, r_image, centers, number)
        # file.write(str(hand_cen_y) + ' ' + str(hand_cen_x) + '\n')
        misc.imsave('zheng_test_results/%s.jpg' % n, result)

    # '''
    #     解析方式二：视频流直接解析
    #     直接读取视频流，并保存在某一个文件夹之中
    # '''
    # # 视频 -> 图像
    # path = "gong.mp4"
    # tracker = Tracker(100, 8, 15, 100)
    # cap = cv2.VideoCapture(path)
    # frame_num = int(cap.get(7))
    # n = 0
    # while True:
    #     ret, frame = cap.read()
    #     if frame is None:
    #         break
    #     image = Image.fromarray(frame)
    #     r_image, out_boxes, out_scores, out_classes, handplace = yolo_test.detect_image(image, frame_num)
    #     centers, number = calc_center(out_boxes, out_classes, out_scores, score_limit=0.6)
    #     tracker, result = trackerDetection(tracker, r_image, centers, number, max_point_distance=20)
    #     # misc.imsave('test_result/gong1_%s.jpg' % n, result)
    #     cv2.imwrite('test_result/gong1_%s.jpg' % n, result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #     n += 1
    # print('Down!')

    # '''
    #     辅助函数
    #     图像文件夹直接变为视频并保存
    # '''
    #
    # # 图像 -> 视频
    # def get_file_names(search_path):
    #     for (dirpath, _, filenames) in os.walk(search_path):
    #         for filename in filenames:
    #             yield filename  # os.path.join(dirpath, filename)
    #
    #
    # def save_to_video(output_path, output_video_file, frame_rate):
    #     # list_files = sorted([int(i.split('_')[-1].split('.')[0]) for i in get_file_names(output_path)])
    #     list_files = sorted([int(i.split('.')[0]) for i in get_file_names(output_path)])
    #     # 拿一张图片确认宽高
    #     img0 = cv2.imread(os.path.join(output_path, '%s.jpg' % list_files[0]))
    #     # print(img0)
    #     height, width, layers = img0.shape
    #     # 视频保存初始化 VideoWriter
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    #     # 核心，保存的东西
    #     for f in list_files:
    #         f = '%s.jpg' % f
    #         # print("saving..." + f)
    #         img = cv2.imread(os.path.join(output_path, f))
    #         videowriter.write(img)
    #     videowriter.release()
    #     cv2.destroyAllWindows()
    #     print('Success save %s!' % output_video_file)
    #     pass
    #
    #
    # # 图片变视频
    # output_path = "test_result"  # 输入图片存放位置
    # output_video_file = 'test_result/video_result/gong_fps20.mp4'  # 输入视频保存位置以及视频名称
    # save_to_video(output_path, output_video_file, 20)
