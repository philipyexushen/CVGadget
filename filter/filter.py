import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


class ImageHandler:
    @staticmethod
    def ShowBoxBlur(imgSrc):
        imgBoxFilter = np.empty(imgSrc.shape, np.uint8)
        cv.boxFilter(imgSrc, -1, (50, 50), imgBoxFilter)
        cv.namedWindow("boxFilter")
        cv.imshow("boxFilter",imgBoxFilter)

    @staticmethod
    def ShowMedianBlur(imgSrc):
        imgMedianBlur = np.empty(imgSrc.shape, np.uint8)
        cv.medianBlur(imgSrc, 5, imgMedianBlur)
        cv.namedWindow("medianBlur")
        cv.imshow("medianBlur",imgMedianBlur)

    @staticmethod
    def ShowLaplaceFilter(imgSrc):
        imgLaplace = np.empty(imgSrc.shape, np.uint8)
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        cv.filter2D(imgSrc, -1, kernel, imgLaplace)
        cv.add(imgSrc, imgLaplace, imgLaplace)

        cv.namedWindow("LaplaceFilter")
        cv.imshow("LaplaceFilter",imgLaplace)

    @staticmethod
    def ShowNonSharpeningMasking(imgSrc):
        imgOut = np.empty(imgSrc.shape, np.uint8)
        cv.GaussianBlur(imgSrc, (7, 7), 0.3, imgOut)

        cv.namedWindow("GaussianBlur")
        cv.imshow("GaussianBlur", imgOut)

        imgSharpen = imgSrc - imgOut
        cv.add(imgSharpen, imgOut, imgOut)

        cv.namedWindow("NonSharpeningMasking")
        cv.imshow("NonSharpeningMasking", imgOut)

    @staticmethod
    def ShowLaplaceMulGradient(imgSrc):
        imgGradient = np.empty(imgSrc.shape, np.uint8)
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        cv.filter2D(imgSrc, -1, kernel, imgGradient)

        imgLaplace = np.empty(imgSrc.shape, np.uint8)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        cv.filter2D(imgSrc, -1, kernel, imgLaplace)

        imgOut = np.empty(imgSrc.shape, np.uint8)
        cv.multiply(imgGradient, imgLaplace, imgOut)
        cv.add(imgOut, imgSrc, imgOut)

        cv.namedWindow("LaplaceMulGradient")
        cv.imshow("LaplaceMulGradient", imgOut)

    @staticmethod
    def CreateLaplaceFrequencyFilterTemplate(P, Q):
        H = np.empty((P, Q, 2), np.float32)
        for i in range(P):
            for j in range(Q):
                H[i, j] = 1 - 4*np.pi**2*((i - P/2)**2 + (j - Q/2)**2)

        cv.normalize(H, H, -255, 255, cv.NORM_MINMAX)
        return H

    @staticmethod
    @jit
    def HandleFFTPerChannel(channel, H):
        P = channel.shape[0]*2
        Q = channel.shape[1]*2
        imgSrcFill = np.empty((P, Q), np.float32)
        oriP = channel.shape[0]
        oriQ = channel.shape[1]
        for i in range(oriP):
            for j in range(oriQ):
                imgSrcFill[i, j] = channel[i, j]
                imgSrcFill[i, j] = imgSrcFill[i, j] if (i + j) % 2 == 0 else -imgSrcFill[i, j]

        cv.normalize(imgSrcFill, imgSrcFill, -1, 1, cv.NORM_MINMAX)
        f = cv.dft(np.float32(imgSrcFill), flags=cv.DFT_COMPLEX_OUTPUT)
        f = np.fft.fftshift(f)
        G = np.fft.ifftshift(H*f)
        fs = cv.idft(G)
        fs = cv.magnitude(fs[:, :, 0], fs[:, :, 1])

        return fs

    @staticmethod
    def ShowFilterLaplace(imgSrc):
        b, g, r = cv.split(imgSrc)
        H = ImageHandler.CreateLaplaceFrequencyFilterTemplate(imgSrc.shape[0] * 2, imgSrc.shape[1] * 2)
        H1 = cv.magnitude(H[:, :, 0], H[:, :, 1])
        cv.normalize(H1, H1, 0, 1, cv.NORM_MINMAX)

        b = ImageHandler.HandleFFTPerChannel(b, H)
        g = ImageHandler.HandleFFTPerChannel(g, H)
        r = ImageHandler.HandleFFTPerChannel(r, H)

        merged = cv.merge((b, g, r))
        imgOut = merged[0:imgSrc.shape[0], 0:imgSrc.shape[1], 0:imgSrc.shape[2]]
        cv.normalize(imgOut, imgOut, 0, 1, cv.NORM_MINMAX)

        cv.namedWindow("src")
        cv.imshow("src", imgSrc)
        cv.namedWindow("merged")
        cv.imshow("merged", imgOut)
        '''
            plt.subplot(411), plt.imshow(merged)
            plt.subplot(412), plt.imshow(b)
            plt.subplot(413), plt.imshow(g)
            plt.subplot(414), plt.imshow(r)
            plt.show()
        '''

    @staticmethod
    @jit
    def CreateHomomorphicFilterTemplate(P, Q):
        H = np.empty((P, Q, 2), np.float32)
        for i in range(P):
            for j in range(Q):
                H[i, j] = 1.75 * (1 - np.exp(-((i - P / 2) ** 2 + (j - Q / 2) ** 2)) / 6400) + 0.25

        return H

    @staticmethod
    def ShowHomomorphicFilter(imgSrc):
        # imgSrc = cv.resize(imgSrc, (int(imgSrc.shape[1]/4), int(imgSrc.shape[0]/4)))
        imgLnSrc = imgSrc
        # 先把范围控制下，不然0值被log以后会出无限值
        cv.normalize(imgLnSrc, imgLnSrc, 1, 255, cv.NORM_MINMAX)
        imgLnSrc = np.float64(imgLnSrc)
        cv.log(imgLnSrc, imgLnSrc)
        b, g, r = cv.split(imgLnSrc)
        H = ImageHandler.CreateHomomorphicFilterTemplate(imgLnSrc.shape[0] * 2, imgLnSrc.shape[1] * 2)
        b = ImageHandler.HandleFFTPerChannel(b, H)
        g = ImageHandler.HandleFFTPerChannel(g, H)
        r = ImageHandler.HandleFFTPerChannel(r, H)

        merged = cv.merge((b, g, r))
        imgOut = merged[0:imgSrc.shape[0], 0:imgSrc.shape[1], 0:imgSrc.shape[2]]
        # 两次归一，是因为出来的大小太大了，exp会出无限值
        cv.normalize(imgOut, imgOut, 0, 1, cv.NORM_MINMAX)
        cv.exp(imgOut, imgOut)
        cv.normalize(imgOut, imgOut, 0, 1, cv.NORM_MINMAX)

        cv.cvtColor(imgSrc, cv.COLOR_BGR2RGB, imgSrc)
        cv.cvtColor(imgOut, cv.COLOR_BGR2RGB, imgOut)
        plt.figure(1), plt.imshow(imgSrc)
        plt.figure(2), plt.imshow(imgOut)
        plt.show()


if __name__ == '__main__':
    imgSrc = cv.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    ImageHandler.ShowHomomorphicFilter(imgSrc)

    cv.waitKey()
    cv.destroyAllWindows()







