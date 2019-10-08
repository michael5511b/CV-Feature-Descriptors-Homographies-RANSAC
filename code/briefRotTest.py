import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import BRIEF



if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    numOfMatches = np.zeros(int(360/10) + 1)

    for theta in range(0, 361, 10):
        H = im1.shape[0]
        W = im1.shape[1]
        rotM = cv2.getRotationMatrix2D((H / 2, W / 2), theta, 1)
        im2 = cv2.warpAffine(im1, rotM, (W, H))
        locs1, desc1 = BRIEF.briefLite(im1)
        locs2, desc2 = BRIEF.briefLite(im2)
        matches = BRIEF.briefMatch(desc1, desc2)
        numOfMatches[int(theta/10)] = len(matches)

    print(numOfMatches)

    plt.bar(range(len(numOfMatches)), numOfMatches)
    plt.show()