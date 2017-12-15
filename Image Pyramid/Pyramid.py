import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import jit


def Pyramid(imgSrc : np.array):
    imgPyrDown = cv.pyrDown(imgSrc)
    imgPyrUp = cv.pyrUp(imgPyrDown)
    imgDiff = imgSrc - imgPyrUp[0:imgSrc.shape[0], 0:imgSrc.shape[1], 0:imgSrc.shape[2]]
    cv.normalize(imgDiff, imgDiff, 0, 255, cv.NORM_MINMAX)

    imgDct = np.empty((imgSrc.shape[0],imgSrc.shape[1], 1), np.float32)

    cv.dct(np.float32(imgSrc[0:imgSrc.shape[0], 0:imgSrc.shape[1], 0 : 1]), imgDct)
    cv.normalize(imgDct, imgDct, 0, 1, cv.NORM_MINMAX)
    imgDct = cv.merge((imgDct, imgDct, imgDct))

    plt.figure(1), plt.title("imgPyrDown"), plt.imshow(imgPyrUp)
    plt.figure(2), plt.title("imgSrc"), plt.imshow(imgSrc)
    plt.figure(3), plt.title("imgPyrUp"), plt.imshow(imgDiff)
    plt.figure(4), plt.title("imgDct"), plt.imshow(imgDct)
    plt.show()


if __name__ == "__main__":
    imgSrc = cv.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    Pyramid(imgSrc)

