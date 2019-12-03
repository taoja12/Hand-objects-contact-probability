"""
Script to help visualize the results of the trained model

Author : Anirudh Vemula
Date : 10th November 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

true_trajectpath = open('true_traject.txt', 'w')
pred_trajectpath = open('pred_trajectpath.txt', 'w')


def plot_trajectories(true_trajs, pred_trajs, obs_length, name):
    """
    Function that plots the true trajectories and the
    trajectories predicted by the model alongside
    params:
    true_trajs : numpy matrix with points of the true trajectories
    pred_trajs : numpy matrix with points of the predicted trajectories
    Both parameters are of shape traj_length x maxNumPeds x 3
    obs_length : Length of observed trajectory
    name: Name of the plot
    """
    traj_length, maxNumPeds, _ = true_trajs.shape

    # Initialize figure
    plt.figure()
    im = plt.imread('/home/tao/projects/cnn_img/tao_test_results/43.jpg')
    implot = plt.imshow(im)

    width = 416
    height = 416

    traj_data = {}
    # For each frame/each point in all trajectories
    for i in range(traj_length):
        pred_pos = pred_trajs[i, :]
        true_pos = true_trajs[i, :]

        # For each pedestrian
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Not a ped
                continue
            elif pred_pos[j, 0] == 0:
                # Not a ped
                continue
            else:
                # If he is a ped
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                if (j not in traj_data) and i < obs_length:
                    traj_data[j] = [[], []]

                if j in traj_data:
                    traj_data[j][0].append(true_pos[j, 1:3])
                    traj_data[j][1].append(pred_pos[j, 1:3])

    for j in traj_data:
        c = np.random.rand(3)
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [p[0] * height for p in true_traj_ped]
        true_y = [p[1] * width for p in true_traj_ped]
        pred_x = [p[0] * height for p in pred_traj_ped]
        pred_y = [p[1] * width for p in pred_traj_ped]

        for k in range(len(true_traj_ped)):
            # print str(n) + ':' + str(true_x[k]) + ',' + str(true_y[k])
            # print str(n) + ':' + str(pred_x[k]) + ',' + str(pred_y[k])

            true_trajectpath.write(str(true_x[k]) + ' ' + str(true_y[k]) + '\n')
            pred_trajectpath.write(str(pred_x[k]) + ' ' + str(pred_y[k]) + '\n')
            # if (k + 1) % 8 == 0:
            #     true_trajectpath.write('\n')
            #     pred_trajectpath.write('\n')

        plt.plot(true_x, true_y, color="green", linestyle='dashed', marker='o')
        plt.plot(pred_x, pred_y, color="red", linestyle='dotted', marker='+')

    plt.savefig('plot/' + name + '.png')
    plt.gcf().clear()
    plt.close()


def main():
    """
    Main function
    """
    f = open('save/my_socialmodel/social_results.pkl', 'rb')
    results = pickle.load(f)
    for i in range(len(results)):
        # print i 59
        name = 'sequence' + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], name)


if __name__ == '__main__':
    main()
