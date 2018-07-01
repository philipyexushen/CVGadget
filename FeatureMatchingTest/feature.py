from common import *
import numpy.linalg as npl

@MethodInformProvider
def SIFT(imgSrc:np.ndarray)->np.ndarray:
    # Scale-invariant feature transform
    # 1. 建立图像尺度空间(或高斯金字塔)，并检测极值点
    kernel = cv.getGaussianKernel(5, np.sqrt(2))
    
    return kernel


@MethodInformProvider
def EigImage(imgSrc:np.ndarray)->(np.ndarray, np.ndarray):
    # 从这个例子中可以发现，矩阵特征值是对特征向量进行伸缩和旋转程度的度量
    # 下面的例子从特征值中砍了一半的特征值，发现图片只是模糊了而已
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    imgSrc = imgSrc[0:512, 0:512]
    U, d, V = npl.svd(imgSrc)
    d[125:] = 0
    D = np.diag(d)

    imgDst = np.dot(np.dot(U, D), V)
    return imgSrc, imgDst


@MethodInformProvider
def PCAImage(imgSrc:np.ndarray)->np.ndarray:
    # Hotelling transform
    assert len(imgSrc.shape) == 3
    imgSrc:np.ndarray = np.float64(imgSrc)
    pic = cv.mean(imgSrc)[:3]

    @jit
    def _Cov(imgSrc:np.ndarray, m:tuple)->np.ndarray:
        h ,w = imgSrc.shape[:2]
        n = len(m)
        cov = np.zeros((n, n), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                T = imgSrc[i, j].reshape(3,1)
                cov += np.dot(T,T.transpose())
        ma = np.array(m)
        return cov / (h*w) - np.dot(ma, ma.transpose()) / (h*w)**2

    CX = _Cov(imgSrc, pic)
    U, d, V = npl.svd(CX)
    print(d)

    @jit
    def _Rebuild(imgSrc:np.ndarray, A:np.ndarray, m:tuple)->np.ndarray:
        imgDst = np.zeros(imgSrc.shape, dtype=np.uint8)

        h, w, c = imgSrc.shape
        Y = np.zeros((h, w, c, 1), dtype=np.float64)
        h, w = imgSrc.shape[:2]
        ma = np.array(m).reshape(3, 1)
        print(ma)

        for i in range(h):
            for j in range(w):
                v = imgSrc[i, j].reshape(3, 1) - ma
                Y[i ,j] = A.dot(v)

        AT:np.ndarray = A.copy()
        AT[2] = 0
        AT = AT.transpose()

        for i in range(h):
            for j in range(w):
                imgDst[i, j] = (AT.dot(Y[i, j]) + ma).reshape(3)

        return imgDst

    return _Rebuild(imgSrc, V, pic)


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