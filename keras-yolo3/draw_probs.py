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
from PIL import Image
from PIL import ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from scipy import misc
from tqdm import tqdm

from KalmanFilterTracker import Tracker
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image


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

    pic_path = "/home/tao/projects/cnn_img/zheng_test/"
    pic_list = os.listdir(pic_path)
    num = len(pic_list)
    tracker = Tracker(100, 8, 15, 100)
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(yolo_test.class_names), 1., 1.)
                  for x in range(len(yolo_test.class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for n in tqdm(range(num)):
        image = Image.open(pic_path + '%s.jpg' % (n + 1))
        r_image, out_boxes, out_scores, out_classes = yolo_test.detect_image(image, num)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in list(enumerate(out_classes)):
            predicted_class = yolo_test.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            print label
            draw = ImageDraw.Draw(image)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if 'pincers' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    pincers_col = [row['pincers'] for row in reader]
                labels = []
                for j in range(len(pincers_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(pincers_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'pen_box' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    pen_box_col = [row['pen_box'] for row in reader]
                labels = []
                for j in range(len(pen_box_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(pen_box_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'plug' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    plug_col = [row['plug'] for row in reader]
                labels = []
                for j in range(len(plug_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(plug_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'cup' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    cup_col = [row['cup'] for row in reader]
                labels = []
                for j in range(len(cup_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(cup_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'tennis_ball' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    ball_col = [row['tennis_ball'] for row in reader]
                labels = []
                for j in range(len(ball_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(ball_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'eraser' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    eraser_col = [row['eraser'] for row in reader]
                labels = []
                for j in range(len(eraser_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(eraser_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            # if 'lotion' in label:
            #     with open("/home/tao/projects/cnns_data/tao_probs.csv", "r") as f:
            #         reader = csv.DictReader(f)
            #         lotion_col = [row['lotion'] for row in reader]
            #     labels = []
            #     for j in range(len(lotion_col)):
            #         label = '{} {:.5f}'.format(predicted_class, float(lotion_col[j]))  # eraser 0.05086
            #         labels.append(label)
            #     label_size = draw.textsize(label, font)
            #     text_origin = np.array([left, top - label_size[1]])
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            #     draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
            #     del draw

            if 'magic_square' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    square_col = [row['magic_square'] for row in reader]
                labels = []
                for j in range(len(square_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(square_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'pen' in label:
                draw = ImageDraw.Draw(image)
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    pen_col = [row['pen'] for row in reader]
                labels = []
                for j in range(len(pen_col)):
                    label = '{} {:.5f}'.format(predicted_class, float(pen_col[j]))  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, labels[n], fill=(0, 0, 0), font=font)  # write text
                del draw

            if 'hand' in label:
                with open("/home/tao/projects/cnns_data/zheng_probs.csv", "r") as f:
                    reader = csv.DictReader(f)
                    hand_col = [row['on_the_way'] for row in reader]
                labels = []
                for j in range(len(hand_col)):
                    label = '{}'.format(predicted_class)  # eraser 0.05086
                    labels.append(label)
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # write text
                del draw

        pointdraw = ImageDraw.Draw(image)
        del pointdraw
        misc.imsave('/home/tao/projects/cnn_img/zheng_test_results/%s.jpg' % (n+1), image)
