import math
import os
import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, Data
import tensorflow as tf
from copy import deepcopy
DELTA = 10
WEIGHTS = [.5, .5]


def linear(B, x):
    return B[0] * x + B[1]


def softmax(object_and_metric, return_sorted=False):
    total = 0
    try:
        for obj in object_and_metric:
            total += math.exp(-math.log(obj[1] + .00001))
        probs = [(i[0], math.exp(-math.log(i[1] + .00001)) / total) for i in object_and_metric]
        total_probs = np.sum([i[1] for i in probs])
    except:
        print('log error on', object_and_metric)
        total_probs = 1
        probs = [(i, 1 / 16) for i in list(zip(list(objects['x_pos']), list(objects['y_pos'])))]

    if abs(1 - total_probs) > 0.001:
        print('Warning! Total probability over metric does not sum to 1. It is', total_probs)

    if return_sorted:
        return sorted(probs, key=lambda x: x[1], reverse=True)
    else:
        return probs


def measure_objects(f, coeff):
    obj_locations = list(zip(list(objects['x_pos']), list(objects['y_pos'])))
    obj_and_metric = []
    for obj in obj_locations:
        error = abs(obj[1] - f(coeff, obj[0]))
        obj_and_metric.append((obj, error))
    return obj_and_metric


# def weigh_probabilities(previous, current):
#     p, c = sorted(previous, key=lambda x: (x[0][0], x[0][1])), sorted(current, key=lambda x: (x[0][0], x[0][1]))
#     probs_p = np.array([prob[1] for prob in p])
#     probs_c = np.array([prob[1] for prob in c])
#     # updated = np.array([probs_p, probs_c])  # np.array([WEIGHTS])
#     updated = tf.matmul(np.array([WEIGHTS]), np.array([probs_p, probs_c]))
#     final = [(p[i][0], updated[0][i]) for i in range(16)]
#     return sorted(final, key=lambda x: x[1], reverse=True)


hand_centers = pd.read_csv("object_choice_prediction/main_files/trial_data/sitting_hand_locations.csv", index_col=0)
objects = pd.read_csv("object_choice_prediction/main_files/trial_data/object_positions.csv", sep=" ")

clip_data = hand_centers[hand_centers['imgs'].str.contains('Tao_Sitting' + "/")]  #                    imgs      x      y
                                                                           # 0      gong/clip1/1.jpg  181.0  231.0
                                                                           # get all frames named gong/...
object_locations = list(zip(list(objects['x_pos']), list(objects['y_pos'])))
# [(271, 308), (165, 319), (213, 257), (226, 373)]

previous_update, latest_update = [(i, 1 / 16) for i in object_locations], [(i, 1 / 16) for i in object_locations]
# [((271, 308), 0), ((165, 319), 0), ((213, 257), 0), ((226, 373), 0)]
# [((271, 308), 0), ((165, 319), 0), ((213, 257), 0), ((226, 373), 0)]

for current_frame in range(1, len(clip_data)):
    frame = clip_data.iloc[current_frame][0]  # gong/clip1/313.jpg
    upper_range = current_frame  # pic number 1-592
    lower_range = max(0, upper_range - DELTA)  # pic_number - 10
    segment_data = clip_data.iloc[lower_range:upper_range + 1]  #               imgs      x      y
                                                                # 0  gong/clip1/1.jpg  181.0  231.0
                                                                # 1  gong/clip1/2.jpg  180.0  231.0
    segment_data = segment_data[abs(segment_data['x'].mean() - segment_data['x']) < 2 * segment_data['x'].std()]
    segment_data = segment_data[abs(segment_data['y'].mean() - segment_data['y']) < 2 * segment_data['y'].std()]

    x = list(segment_data['x'])
    y = list(segment_data['y'])

    if len(x) == 0:
        continue

    mydata = Data(x, y)

    f = linear
    mod = Model(linear)
    myodr = ODR(mydata, mod, beta0=[0, 2])
    # print(myodr)
    res = myodr.run()
    coeff = res.beta
    obj_and_metric = measure_objects(f, coeff)

    obj_locations = list(zip(list(objects['x_pos']), list(objects['y_pos'])))
    for obj in obj_locations:
        error = abs(obj[1] - f(coeff, obj[0]))
        obj_and_metric.append((obj, error))

    if x[-1] >= x[0]:
        hand_movement_dir = "right"
    else:
        hand_movement_dir = "left"
    for i in range(len(obj_and_metric)):
        obj_loc = obj_and_metric[i][0]
        if ((obj_loc[0] < x[-1]) and hand_movement_dir == "left") or \
                ((obj_loc[0] > x[-1]) and hand_movement_dir == "right"):
            pass
        else:
            obj_and_metric[i] = (obj_and_metric[i][0], np.inf)

    latest_update = softmax(obj_and_metric, return_sorted=True)

    p, c = sorted(previous_update, key=lambda x: (x[0][0], x[0][1])), sorted(latest_update, key=lambda x: (x[0][0], x[0][1]))
    probs_p = np.array([prob[1] for prob in p])
    probs_c = np.array([prob[1] for prob in c])
    updated1 = np.array([WEIGHTS])
    print("updated1", updated1)
    updated2 = np.array([probs_p, probs_c])
    print("updated2", updated2)