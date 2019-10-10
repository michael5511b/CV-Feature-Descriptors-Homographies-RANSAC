import numpy as np
import cv2
import os
from planarH import computeH
from matplotlib import pyplot as plt


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation

    '''

    #############################
    # We want to get the new homography that eliminatesthe effect of the intrinsic parameters
    # by pre-multiplying the estimated homography by the inverse of the intrinsic matrix
    # Follow instructions in the Prince textbook
    H_new = np.dot(np.linalg.inv(K), H)
    U, L, Vt = np.linalg.svd(H_new[:, 0:2], full_matrices=True)
    w = np.dot(U, np.array([[1, 0], [0, 1], [0, 0]]))
    w = np.dot(w, Vt)

    w = np.concatenate((w, np.cross(w[:, 0], w[:, 1]).reshape((3,1))),axis=1)

    if np.linalg.det(w) == -1:
        w[:, 2] = w[:, 2] * (-1)

    sum = 0
    for i in range(3):
        for j in range(2):
            sum += H_new[i, j] / w[i, j]
    scale = sum / 6

    R = w
    t = (H_new[:, 2] / scale).T


    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    extrinsic = np.concatenate((R, t), axis=1)
    intrin_extrin = np.dot(K, extrinsic)
    x = np.dot(intrin_extrin, W)

    proj = np.zeros((2, W.shape[1]))
    for i in range(x.shape[1]):
        # remember scaling
        proj[0, i] = int(x[0, i] / x[2, i])
        proj[1, i] = int(x[1, i] / x[2, i])

    return proj


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')
    book_im = cv2.imread('../data/prince_book.jpeg')
    # plt.imshow(im, cmap='gray')
    # plt.show()
    #############################
    # TO DO ...
    # perform required operations and plot sphere

    # Four corners of book in real life
    W = np.array([[0, 18.2, 18.2, 0], [0, 0, 26, 26], [0, 0, 0, 0]])
    # Four corners of the book on image
    X = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    # Intrinsic matrix for image
    K = np.array([[3043.72, 0, 1196], [0, 3043.72, 1604], [0, 0, 1]])


    H = computeH(X, W[0:2, :])
    R, t = compute_extrinsics(K, H)

    # Open sphere file
    file = open('../data/sphere.txt', "r")
    str = file.read()
    str_array = str.split('\n')
    sphere_x = str_array[0].split('  ')
    sphere_y = str_array[1].split('  ')
    sphere_z = str_array[2].split('  ')
    tmp_coord_list = []
    for i in range(1, len(sphere_x)):
        tmp_coord_list.append([float(sphere_x[i]), float(sphere_y[i]), float(sphere_z[i])])

    W_sphere = np.array(tmp_coord_list).T
    W_sphere = np.concatenate((W_sphere, np.ones((1, W_sphere.shape[1]))), axis=0)

    # With the sphere coordinates, project extrinsics!
    proj_sphere = project_extrinsics(K, W_sphere, R, t.reshape(3, 1)) + (np.array((320, 650))).T.reshape(2, 1)

    fig = plt.figure()
    plt.imshow(im)
    plt.plot(proj_sphere[0, :], proj_sphere[1, :], 'y.', markersize=2)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)