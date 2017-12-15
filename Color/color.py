import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def PseudoColor(imsSrc):
    pseudoColor = [(90 + i, 40 + 2*i, 30 + 2*i) for i in range(64)]
    sz = len(pseudoColor)
    h, w = imgSrc.shape[0], imgSrc.shape[1]
    imgOut = np.empty((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            gary = imgSrc[i, j]
            color = pseudoColor[gary % sz]
            imgOut[i, j] = color

    return imgOut


if __name__ == "__main__":
    imgSrc = cv.imread("E:/Users/Administrator/pictures/Test/origin3.jpg", cv.IMREAD_GRAYSCALE)
    imgOut = PseudoColor(imgSrc)

    plt.figure(1), plt.imshow(imgSrc, cmap="gray")
    plt.figure(2), plt.imshow(imgOut)
    plt.show()