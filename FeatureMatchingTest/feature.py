from common import *
import numpy.linalg as npl

@MethodInformProvider
def SIFT(imgSrc:np.ndarray)->np.ndarray:
    # Scale-invariant feature transform
    pass


@MethodInformProvider
def EigImage(imgSrc:np.ndarray)->(np.ndarray, np.ndarray):
    # 从这个例子中可以发现，矩阵特征值是对特征向量进行伸缩和旋转程度的度量
    # 下面的例子从特征值中砍了一半的特征值，发现图片只是模糊了而已
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    imgSrc = imgSrc[0:512, 0:512]
    U, d, V = npl.svd(imgSrc)
    D = np.diag(d)
    D[125:] = 0

    imgDst = np.dot(np.dot(U, D), V)
    return imgSrc, imgDst


@MethodInformProvider
def CornerHarris(imgSrc:np.ndarray, alpha)->np.ndarray:
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    gx = cv.Sobel(np.float32(imgSrc), -1, 1, 0)
    gy = cv.Sobel(np.float32(imgSrc), -1, 0, 1)

    A, B, C = gx**2, gy**2, gx*gy
    A = cv.GaussianBlur(A, (5, 5), 1)
    B = cv.GaussianBlur(B, (5, 5), 1)
    C = cv.GaussianBlur(C, (5, 5), 1)

    imgDst = np.abs(A*B - C**2 - alpha*((A + C)**2))
    maxM = np.max(imgDst)
    imgDst = np.where(imgDst < 0.01*maxM, 0, imgDst)
    return imgDst