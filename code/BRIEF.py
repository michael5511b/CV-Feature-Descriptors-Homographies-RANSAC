import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
np.set_printoptions(threshold=sys.maxsize)

def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''
    
    #############################
    # TO DO ...
    # Generate testpattern here

    # From the "BRIEF: Binary robust independent elementary features" paper (section 3.2 GII)
    # It mentioned that if the test set  are sampled from an isotropic Gaussian distribution.
    # it will have the best performance
    # From paper, sigma = patch size / 5 has the best results
    # Set mean (mid) point to patch_size / 2 (which is 9/2) so it is a normal distribution from the
    # middle of the patch
    # 4 columns, holding [x1 x2 y1 y2]
    gauss_distr = np.random.normal(loc=patch_width / 2, scale=patch_width / 5, size=(nbits, 4))

    # Round and convert the values to int, so that they become actual coordinates
    compare = np.round(gauss_distr).astype(int)

    # clip all the values within
    compare = np.clip(compare, 0, patch_width - 1)

    # Seperate the X and Y coordinates into two separate arrays
    compX2D = compare[:, 0:2]
    compY2D = compare[:, 2:4]

    # ravel converts the 2D coordinates into linear indices
    # We can later use np.unravel_index to convert it back to 2D
    compareX = np.ravel_multi_index(compX2D.T, (9, 9))
    compareY = np.ravel_multi_index(compY2D.T, (9, 9))

    return compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the 
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.
    
    
    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''
    
    ##############################
    # TO DO ...
    # compute locs, desc here
    locs = []
    for i in range(locsDoG.shape[0]):
        if locsDoG[i][0] - k//2 >= 0 and locsDoG[i][0] + k//2 <= im.shape[1] - 1 and\
                locsDoG[i][1] - k//2 >= 0 and locsDoG[i][1] +k//2 <= im.shape[0] - 1:
            locs.append(locsDoG[i, :])
    locs = np.array(locs)

    desc = np.zeros((locs.shape[0], compareX.shape[0])).astype(int)

    for i in range(locs.shape[0]):
        x = locs[i][1]
        y = locs[i][0]
        patch = im[x - 4: x + 5, y - 4: y + 5]

        for j in range(compareX.shape[0]):
            x1, y1 = np.divmod(compareX[j], k)
            x2, y2 = np.divmod(compareY[j], k)

            if patch[x1, y1] < patch[x2, y2]:
                desc[i, j] = 1
                
    return locs, desc


def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    
    ###################
    # TO DO ...

    # For BRIEF computation, convert to gray scale of one channel
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255

    test_pattern_file = '../results/testPattern.npy'
    if os.path.isfile(test_pattern_file):
        # load from file if exists
        compareX, compareY = np.load(test_pattern_file)
    else:
        # produce and save patterns if not exist
        compareX, compareY = makeTestPattern(patch_width=9, nbits=256)
        if not os.path.isdir('../results'):
            os.mkdir('../results')
        np.save(test_pattern_file, [compareX, compareY])

    locsDoG, gaussian_pyramid = DoGdetector(im)
    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, 9, [-1,0,1,2,3,4], compareX, compareY)

    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1 / (d2 + 1e-10) # 1e-10 prevents the denominator to be 0
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')

    # print(matches)
    for i in range(matches.shape[0]):

        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')

    plt.savefig('matches.png')
    plt.show()    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    
    # test briefLite
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # locs, desc = briefLite(im)
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.plot(locs[:,0], locs[:,1], 'r.')
    #
    # plt.draw()
    #
    # plt.waitforbuttonpress(0)
    # plt.close(fig)

    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')

    # im1 = cv2.imread('../data/incline_L.png')
    # im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    # print(desc2.shape)
    # print(desc2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
