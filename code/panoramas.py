import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
from matplotlib import pyplot as plt


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################

    out_size = (im1.shape[1] + 200, im1.shape[0])

    im2warpped = cv2.warpPerspective(im2, H2to1, out_size)
    # cv2.imwrite('../results/6_1_left.jpg', im1)
    # cv2.imwrite('../results/6_1_right.jpg', im2)
    # cv2.imwrite('../results/6_1_right_warped.jpg', im2warpped)

    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if np.array_equal(im1[i, j], [0, 0, 0]) and np.array_equal(im2warpped[i, j], [0, 0, 0]):
                im2warpped[i, j] = [0, 0, 0]
            else:
                if np.array_equal(im2warpped[i, j], [0, 0, 0]):
                    im2warpped[i, j] = im1[i, j]
                else:
                    b1, g1, r1 = im1[i, j]
                    b2, g2, r2 = im2warpped[i, j]
                    im2warpped[i, j] = [max(b1, b2), max(g1, g2), max(r1, r2)]

    pano_im = im2warpped

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    ######################################
    H = im2.shape[0]
    W = im2.shape[1]

    # Four corners of im2
    p1 = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]])
    p2 = np.zeros((4, 2))
    for i in range(4):
        p2[i, 0] = np.dot(H2to1[0], p1[i]) / np.dot(H2to1[2], p1[i])
        p2[i, 1] = np.dot(H2to1[1], p1[i]) / np.dot(H2to1[2], p1[i])

    # Use those four corner's new coordinates after warp to determine the size of the new image
    W_pan_1 = int(np.max(p2[:, 0]))
    H_pan_1 = int(np.max(p2[:, 1]) - np.min(p2[:, 1]))

    # take W of im2 as input, warp everything else
    W_pan = W
    H_pan = int((W / W_pan_1) * H_pan_1)

    # set up fitting(scaling / translation) matrix M
    scale1 = W_pan / W_pan_1
    scale2 = H_pan / H_pan_1
    trans1 = 0
    trans2 = -np.min(p2[:, 1]) * scale2
    # only includes scaling and translation
    # [s1 0  t1
    #  0  s2 t2
    #  0  0  1 ]
    M = np.array([[scale1, 0, trans1], [0, scale2, trans2], [0, 0, 1]])

    out_size = (W_pan, H_pan)

    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), out_size)

    # cv2.imwrite('../results/6_2_left.jpg', im1)
    # cv2.imwrite('../results/6_2_right.jpg', im2)
    # cv2.imwrite('../results/6_2_left_warped.jpg', warp_im1)
    # cv2.imwrite('../results/6_2_right_warped.jpg', warp_im2)

    for i in range(warp_im1.shape[0]):
        for j in range(warp_im1.shape[1]):
            if np.array_equal(warp_im1[i, j], [0, 0, 0]) and np.array_equal(warp_im2[i, j], [0, 0, 0]):
                warp_im2[i, j] = [0, 0, 0]
            else:
                if np.array_equal(warp_im2[i, j], [0, 0, 0]):
                    warp_im2[i, j] = warp_im1[i, j]
                else:
                    b1, g1, r1 = warp_im1[i, j]
                    b2, g2, r2 = warp_im2[i, j]
                    warp_im2[i, j] = [max(b1, b2), max(g1, g2), max(r1, r2)]

    pano_im = warp_im2
    
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/6_3_stitched.jpg', pano_im)

    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    #########################################################
    # # p6.1 and p6.2 needed code
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save("../results/q6_1.npy", H2to1)
    #########################################################
    # # p6.1
    # pano_im = imageStitching(im1, im2, H2to1)
    # cv2.imshow('panoramas', pano_im)
    # cv2.imwrite('../results/6_1_stitched.jpg', pano_im)
    #########################################################
    # # p6.2
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # cv2.imshow('panoramas', pano_im)
    # cv2.imwrite('../results/6_2_stitched.jpg', pano_im)
    #########################################################
    generatePanaroma(im1, im2)