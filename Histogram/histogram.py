# coding=utf-8
import cv2 as cv
import numpy as np


def get_max_value_in_histogram(mat):
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(mat)
    return max_val


def make_HShistogram(mat):
    nMaxHistogram = get_max_value_in_histogram(mat)
    scale = 4

    imgHSHistogram = np.zeros((180*scale, 256*scale,3), np.uint8)
    for h in range(180):
        for s in range(256):
            val = mat[h, s]
            intensity = round(val*255/nMaxHistogram)
            cv.rectangle(imgHSHistogram,
                         (h*scale, s*scale),
                         ((h+1)*scale-1, (s+1)*scale-1),
                         (intensity, intensity, intensity, intensity),
                         cv.FILLED)

    return imgHSHistogram


class Int(int):
    def __init__(self, n):
        int.__init__(n)

    def update(self, n):
        return self



if __name__ == '__main__':

    imgSrc = cv.imread("E:/Users/Administrator/pictures/Test/user.jpg")

    vecOutChannels = [None]*3
    vecChannels = cv.split(imgSrc)

    vecOutChannels[0] = cv.equalizeHist(vecChannels[0])
    vecOutChannels[1] = cv.equalizeHist(vecChannels[1])
    vecOutChannels[2] = cv.equalizeHist(vecChannels[2])

    imgDst = cv.merge(vecOutChannels)

    cv.namedWindow("Source")
    cv.imshow("Source", imgSrc)

    cv.namedWindow("Histogram")
    cv.imshow("Histogram", imgDst)

    imgHSVSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2HSV)
    imgHSVDst = cv.cvtColor(imgDst, cv.COLOR_RGB2HSV)

    matSrcHistogram = cv.calcHist([imgHSVSrc], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.namedWindow("Source Histogram")
    cv.imshow("Source Histogram", make_HShistogram(matSrcHistogram))
    cv.normalize(matSrcHistogram, matSrcHistogram, 0, 1, cv.NORM_MINMAX)

    matDstHistogram = cv.calcHist([imgHSVDst], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.namedWindow("Destination Histogram")
    cv.imshow("Destination Histogram", make_HShistogram(matDstHistogram))
    cv.normalize(matDstHistogram, matDstHistogram, 0, 1, cv.NORM_MINMAX)

    for i in range(6):
        """
        HISTCMP_CORREL = 0
        HISTCMP_CHISQR = 1
        HISTCMP_INTERSECT = 2
        HISTCMP_BHATTACHARYYA = 3
        HISTCMP_HELLINGER = 3
        HISTCMP_CHISQR_ALT = 4
        HISTCMP_KL_DIV = 5
        """
        #print(cv.compareHist(matSrcHistogram, matSrcHistogram, i))
        print(cv.compareHist(matSrcHistogram, matDstHistogram, i))

    cv.waitKey(0)
    cv.destroyAllWindows()