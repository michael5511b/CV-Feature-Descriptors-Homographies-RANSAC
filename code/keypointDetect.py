import numpy as np
import cv2
import imageio


def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels = [-1,0,1,2,3,4]):
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # imageio.imwrite('Q12.png', im_pyramid)
    # cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    
    X, Y, L = gaussian_pyramid.shape
    DoG_pyramid = np.zeros((X, Y, L - 1))
    ################
    # TO DO ...
    # compute DoG_pyramid here
    # Research LoG vs DoG
    for i in range(len(levels) - 1):
        DoG_pyramid[:, :, i] = gaussian_pyramid[:, :, i + 1] - gaussian_pyramid[:, :, i]
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''

    
    ##################
    # TO DO ...
    # Compute principal curvature here

    H, W, K = DoG_pyramid.shape
    principal_curvature = np.zeros((H, W, K))
    for k in range(K):
        D = DoG_pyramid[:, :, k]
        # cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
        # if ddepth = -1, means using the same depth as the original image
        # H (Hessian of the DoG function) = [Dxx, Dxy, Dyx, Dyy]
        ksize = 5
        # Dx = cv2.Sobel(D, -1, 1, 0, ksize)
        # Dy = cv2.Sobel(D, -1, 0, 1, ksize)
        Dxx = cv2.Sobel(D, -1, 2, 0, ksize)
        Dxy = cv2.Sobel(D, -1, 1, 1, ksize)
        Dyx = cv2.Sobel(D, -1, 1, 1, ksize)
        Dyy = cv2.Sobel(D, -1, 0, 2, ksize)
        for i in range(int(H)):
            for j in range(int(W)):
                # R = (Trace(H) ^ 2) / Det(H)
                hess = np.array([[Dxx[i, j], Dxy[i, j]], [Dyx[i, j], Dyy[i, j]]])
                t = np.trace(hess)
                det = np.linalg.det(hess)
                # Prevent det being 0
                if det == 0:
                    det = 0.000001
                principal_curvature[i, j, k] = (t ** 2) / det
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    
    ##############
    #  TO DO ...
    # Compute locsDoG here
    # This is one way to do it:
    """
    H, W, K = DoG_pyramid.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            for k in range(K):
                # if statement not finished yet
                if(DoG_pyramid[i, j, k] > DoG_pyramid[i - 1, j - 1, k] and DoG_pyramid[i, j, k] > DoG_pyramid[i - 1, j, k]
                        and DoG_pyramid[i, j, k] > DoG_pyramid[i - 1, j + 1, k].....):
    """

    # This is another way that requires a lot less loops!
    # We can use dilation and erosion to get the local max and min respectively

    # First we get the kernel for dilation and erosion
    # Because we are searching the surrounding 8 pixels, so our erosion and dilation will be done with a 3x3 kernel
    # MORPH_RECT makes a rectangular kernel, oppose to MORPH_CROSS, MORPH_ELLIPSE
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Dilation: The value of the output pixel (the output will be at the middle of the 3x3, "middle") is the maximum
    #          value in the coverage of the kernel (3x3 space).
    dilated_DoG = cv2.dilate(DoG_pyramid, kernel_3x3)
    # Only when the maximum element is in the middle of the 3x3 kernel, where the original and the dilated pixel
    # at the same location will have the same value. Thus only the local Extrema (max in this case) will be stored in
    # the array max
    max = np.where(DoG_pyramid == dilated_DoG)

    # max = np.array([max[0], max[1], max[2]]).transpose().tolist()
    # print(len(max))
    # Use loop to check if the pixel is also the max of neighbor scales and if it fits within threshold
    check = 0
    for i in range(len(max[0])):
        x = max[0][i]
        y = max[1][i]
        z = max[2][i]
        # print(np.array([[x, y, z]]))

        if z == 0:
            if DoG_pyramid[x, y, z] > DoG_pyramid[x, y, z + 1] and abs(DoG_pyramid[x, y, z]) > th_contrast and\
                    principal_curvature[x, y, z] < th_r:
                if check == 0:
                    check += 1
                    locsDoG = np.array([[x, y, z]])
                else:
                    locsDoG = np.concatenate((locsDoG, np.array([[x, y, z]])), axis=0)
        elif z == 4:
            if DoG_pyramid[x, y, z] > DoG_pyramid[x, y, z - 1] and abs(DoG_pyramid[x, y, z]) > th_contrast and\
                    principal_curvature[x, y, z] < th_r:
                if check == 0:
                    check += 1
                    locsDoG = np.array([[x, y, z]])
                else:
                    locsDoG = np.concatenate((locsDoG, np.array([[x, y, z]])), axis=0)
        else:
            if DoG_pyramid[x, y, z] > DoG_pyramid[x, y, z - 1] and \
                    DoG_pyramid[x, y, z] > DoG_pyramid[x, y, z + 1] and\
                    abs(DoG_pyramid[x, y, z]) > th_contrast and principal_curvature[x, y, z] < th_r:
                if check == 0:
                    check += 1
                    locsDoG = np.array([[x, y, z]])
                else:

                    locsDoG = np.concatenate((locsDoG, np.array([[x, y, z]])), axis=0)


    X = locsDoG[:, 0]
    Y = locsDoG[:, 1]
    Z = locsDoG[:, 2]
    G = np.zeros((X.shape[0],3),dtype=int)
    G[:, 0] = Y
    G[:, 1] = X
    G[:, 2] = Z
    locsDoG = G

    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    gaussian_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, Dog_levels = createDoGPyramid(gaussian_pyramid, levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, Dog_levels, principal_curvature, th_contrast, th_r)

    return locsDoG, gaussian_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)