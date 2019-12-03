# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import colorsys
import csv
import os
import random
from timeit import default_timer as timer

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from scipy import misc
from tqdm import tqdm

from KalmanFilterTracker import Tracker  # 加载卡尔曼滤波函数
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

tracknum = []
allhand = 0
framenum = 0
# file = open('hand_center.txt', 'w')
file = open('zheng_test_center/hand_center.txt', 'w')
file_ball = open('zheng_test_center/tennis_ball_center.txt', 'w')
file_eraser = open('zheng_test_center/eraser_center.txt', 'w')
file_lotion = open('zheng_test_center/lotion_center.txt', 'w')
file_square = open('zheng_test_center/square_center.txt', 'w')
file_pen = open('zheng_test_center/pen_center.txt', 'w')
file_box = open('zheng_test_center/pen_box_center.txt', 'w')
file_pincers = open('zheng_test_center/pincers_center.txt', 'w')
file_plug = open('zheng_test_center/plug_center.txt', 'w')
file_cup = open('zheng_test_center/cup_center.txt', 'w')


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_20000.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, num):
        start = timer()
        image = image.resize(self.model_image_size)
        handplace = []

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)
            temppoint = [center_x, center_y]  # current hand position

            if 'pincers' in label:
                pincers_cen_x = int((left + right) / 2)
                pincers_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_pincers.write(str(pincers_cen_x) + ' ' + str(pincers_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'pen' in label:
                pen_cen_x = int((left + right) / 2)
                pen_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_pen.write(str(pen_cen_x) + ' ' + str(pen_cen_y) + '\n')
                print(str(pen_cen_x) + ' ' + str(pen_cen_y))
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'pen_box' in label:
                box_cen_x = int((left + right) / 2)
                box_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_box.write(str(box_cen_x) + ' ' + str(box_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'plug' in label:
                plug_cen_x = int((left + right) / 2)
                plug_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_plug.write(str(plug_cen_x) + ' ' + str(plug_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'cup' in label:
                cup_cen_x = int((left + right) / 2)
                cup_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_cup.write(str(cup_cen_x) + ' ' + str(cup_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'tennis_ball' in label:
                ball_cen_x = int((left + right) / 2)
                ball_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_ball.write(str(ball_cen_x) + ' ' + str(ball_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'eraser' in label:
                eraser_cen_x = int((left + right) / 2)
                eraser_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_eraser.write(str(eraser_cen_x) + ' ' + str(eraser_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'lotion' in label:
                lotion_cen_x = int((left + right) / 2)
                lotion_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_lotion.write(str(lotion_cen_x) + ' ' + str(lotion_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'magic_square' in label:
                square_cen_x = int((left + right) / 2)
                square_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file_square.write(str(square_cen_x) + ' ' + str(square_cen_y) + '\n')
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if 'hand' in label:
                hand_cen_x = int((left + right) / 2)
                hand_cen_y = int((top + bottom) / 2)
                tracknum.append(temppoint)
                file.write(str(hand_cen_x) + ' ' + str(hand_cen_y) + '\n')
                print(str(hand_cen_x) + ' ' + str(hand_cen_y))
            #     if top - label_size[1] >= 0:
            #         text_origin = np.array([left, top - label_size[1]])
            #     else:
            #         text_origin = np.array([left, top + 1])
            #     # My kingdom for a good redistributable image drawing library.
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            #     del draw
            # pointdraw = ImageDraw.Draw(image)
            # del pointdraw

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
            del draw
        # pointdraw = ImageDraw.Draw(image)
        # for i in tracknum:
        # print(len(tracknum))
        # pointdraw.ellipse((i[0] - 5, i[1] - 5, i[0] + 5, i[1] + 5), (0, 255, 0), (0, 255, 0), 2)
        # del pointdraw
        # end = timer()
        # print(end - start)
        return image, out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()


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
        r_image, out_boxes, out_scores, out_classes = yolo_test.detect_image(image, num)
        centers, number, tracknum = calc_center(out_boxes, out_classes, out_scores, score_limit=0.5)
        tracker, result = trackerDetection(tracker, r_image, centers, number)
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
