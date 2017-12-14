#-*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.font_manager import FontProperties
import time

def MethodInformation(method):
    def __decorator(*args, **kwargs):
        t0 = time.clock()
        print(f"[Call {method.__name__}]")
        ret = method(*args, **kwargs)
        t1 = time.clock()
        print(f"[Method {method.__name__} take {t1 - t0}s to execute]")
        print(f"[Out {method.__name__}]")
        return ret
    return __decorator

kernelB = [np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8),
               np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8),
               np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=np.uint8),
               np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
               np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
               np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
               np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8),
               np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
              ]

kernelBInV = [np.array([[1, 1, 1],[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
                  np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8),
                  np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8),
                  np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8),
                  np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8),
                  np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8),
                  np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8),
                  np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
                  ]

@jit
def Not(imgSrc:np.ndarray)->np.ndarray:
    imgDst = np.zeros(imgSrc.shape, dtype=np.uint8)
    imgDst += 255
    return imgDst - imgSrc

@jit
def DoOperationPerChannel(loc:tuple, sz:tuple, src:np.ndarray, dst:np.ndarray, kernel:np.ndarray, ksz:tuple, op)->None:
    y, x = loc[0], loc[1]
    ly, hy, lx, hx, lc, hc = sz[0], sz[1], sz[2], sz[3], sz[4], sz[5]
    ky, kx = ksz[0], ksz[1]

    for c in range(lc, hc):
        s0 = src[y, x, c]
        for i, ki in zip(range(y, y + ky), range(ky)):
            for j, kj in zip(range(x, x + kx), range(kx)):
                if ly <= i < hy and lx <= j < hx:
                    r = int(src[i ,j, c]) - kernel[ki, kj]
                    s0 = op(s0, r if r >= 0 else 0)
        dst[y, x, c] = s0

@jit
def MyDilateByFlatStructure(imgSrc:np.ndarray, kernel:np.ndarray)->np.ndarray:
    # 自己写的膨胀函数的版本，太慢了，直接用库的吧
    assert kernel.shape[0] %2 == 1
    assert kernel.shape[1] %2 == 1
    assert len(kernel.shape) == 2
    imgDst = np.empty(imgSrc.shape, np.uint8)

    h, w, channel = imgSrc.shape[0], imgSrc.shape[1],imgSrc.shape[2]
    ky, kx = kernel.shape[0], kernel.shape[1]
    for i in range(h):
        for j in range(w):
            DoOperationPerChannel((i, j), (0, h, 0, w, 0, channel),imgSrc, imgDst,kernel,(ky, kx), max)

    return imgDst

@jit
def MyErodeByFlatStructure(imgSrc:np.ndarray, kernel:np.ndarray)->np.ndarray:
    # 自己写的腐蚀函数的版本，太慢了，直接用库的吧
    assert kernel.shape[0] % 2 == 1
    assert kernel.shape[1] % 2 == 1
    assert len(kernel.shape) == 2
    imgDst = np.empty(imgSrc.shape, np.uint8)
    tKernel = np.transpose(kernel)

    h, w, channel = imgSrc.shape[0], imgSrc.shape[1], imgSrc.shape[2]
    for i in range(h):
        for j in range(w):
            DoOperationPerChannel((i, j), (0, h, 0, w, 0, channel), imgSrc, imgDst, tKernel, min)

    return imgDst

@jit
def Dilate(imgSrc:np.ndarray, kernel:np.ndarray, count = 1)->np.ndarray:
    if count is 0:
        return imgSrc.copy()

    imgRes1 = imgSrc
    imgRes2 = np.empty(imgRes1.shape, dtype=np.uint8)
    for i in range(count):
        cv.dilate(imgRes1, kernel, imgRes2)
        imgRes1 = imgRes2
    return imgRes1

@jit
def Erode(imgSrc:np.ndarray, kernel:np.ndarray, count = 1)->np.ndarray:
    if count is 0:
        return imgSrc.copy()

    imgRes1 = imgSrc
    imgRes2 = np.empty(imgRes1.shape, dtype=np.uint8)
    for i in range(count):
        cv.erode(imgRes1, kernel, imgRes2)
        imgRes1 = imgRes2
    return imgRes1

@jit
def MyHitMiss(imgSrc:np.ndarray, hit:np.ndarray, miss:np.ndarray)->np.ndarray:
    imgDst1 = Erode(imgSrc, hit)
    imgDst2 = Not(Dilate(imgSrc, cv.transpose(miss)))
    imgDst = np.empty(imgSrc.shape, dtype=np.uint8)
    h, w = imgSrc.shape
    for (i, L1, L2) in zip(range(h), imgDst1, imgDst2):
        for (j, item1, item2) in zip(range(w), L1, L2):
            imgDst[i, j] = min(item1, item2)

    return imgDst

@jit
def Thining(imgSrc:np.ndarray)->np.array:
    imgRes1 = imgSrc.copy()
    sz:int = len(kernelB)
    for i in range(sz):
        imgRes1 = imgRes1 - MyHitMiss(imgRes1, kernelB[i], kernelBInV[i])

    return imgRes1

def Thicking(imgSrc:np.ndarray)->np.ndarray:
    '''
    imgRes1 = imgSrc.copy()
    sz:int = len(kernelB)
    h, w = imgSrc.shape[:2]

    for i in range(sz):
        imgRes2 = MyHitMiss(imgRes1, kernelBInV[i], kernelB[i])
        for y in range(h):
            for x in range(w):
                imgRes2[y, x] = max(imgRes2[y, x], imgRes1[y, x])
        imgRes1 = imgRes2
    '''

    # 比较常用的算法是先对A求补给求细化，再求结果的补集
    return Not(Thining(Not(imgSrc)))

@jit
def Skeletons(imgSrc:np.ndarray, kernel:np.ndarray)->np.ndarray:
    K = 0
    Zeros = np.zeros(imgSrc.shape, dtype=np.uint8)
    imgTmp = imgSrc
    while True:
        imgTmp = Erode(imgTmp, kernel)
        if (imgTmp == Zeros).all():
            break
        K += 1

    imgDst = np.zeros(imgSrc.shape, dtype=np.uint8)
    for i in range(K):
        imgErode = Erode(imgSrc, kernel, i + 1)
        imgDst = cv.bitwise_or(imgErode - Dilate(Erode(imgErode, kernel), kernel), imgDst)

    return imgDst

@jit
def RebuildMorphOpen(imgSrc:np.ndarray, kernel:np.ndarray, count:int = 1) -> np.ndarray:
    imgRes1 = Erode(imgSrc, kernel, count)
    h, w, c = imgSrc.shape

    while True:
        imgRes2 = Dilate(imgRes1, kernel, 1)

        flag = True
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    r = min(imgRes2[i, j, k], imgSrc[i, j, k])
                    imgRes2[i, j, k] = r
                    if r != imgRes1[i, j, k]:
                        flag = False
        if flag:
            return imgRes2
        imgRes1 = imgRes2

@jit
def RebuildMorphClose(imgSrc:np.ndarray, kernel:np.ndarray, count:int = 1) -> np.ndarray:
    imgRes1 = Dilate(imgSrc, kernel, count)
    h, w, c = imgSrc.shape

    while True:
        imgRes2 = Erode(imgRes1, kernel, 1)
        flag = True
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    r = max(imgRes2[i, j, k], imgSrc[i, j, k])
                    imgRes2[i, j, k] = r
                    if r != imgRes1[i, j, k]:
                        flag = False
        if flag:
            return imgRes2
        imgRes1 = imgRes2

@jit
def FillHole(imgSrc:np.ndarray, kernel:np.ndarray)->np.ndarray:
    h, w, = imgSrc.shape

    imgSrcInv = Not(imgSrc)
    imgRes1 = Dilate(imgSrc, kernel) - Erode(imgSrc, kernel)

    while True:
        imgRes2 = Dilate(imgRes1, kernel, 1)
        flag = True
        for i in range(h):
            for j in range(w):
                r = min(imgRes2[i, j], imgSrcInv[i, j])
                imgRes2[i, j] = r
                if r != imgRes1[i, j]:
                    flag = False
        if flag:
            return imgRes2
        imgRes1 = imgRes2

@jit
def ConvexHull(imgSrc:np.ndarray)->np.ndarray:
    # 注：此方法很难收敛，还是用Graham扫描法好一点
    # 凸包的四个hit结构元
    kernel = [np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.uint8),
              np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
              np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8),
              np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)]

    # 凸包的四个miss结构元
    kernelInv = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
                 np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
                 np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
                 np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)]

    imgDst = np.zeros(imgSrc.shape, dtype=np.uint8)
    h, w = imgSrc.shape
    for i in range(4):
        imgRes1 = imgSrc
        counter = 0
        while True:
            imgRes2 = MyHitMiss(imgRes1, kernel[i], kernelInv[i])
            flag = bool(True)
            for y in range(h):
                for x in range(w):
                    r = max(imgRes2[y, x], imgSrc[y, x])
                    imgRes2[y, x] = r
                    if r != imgRes1[y, x]:
                        flag = False
            # 因为太难收敛了，来个阈值限制下
            if flag or counter > 3:
                imgDst = cv.bitwise_or(imgDst, imgRes2)
                break
            imgRes1 = imgRes2
            counter += 1

    return imgDst

if __name__ == "__main__":
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/Calculator/skeletons.jpg")
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    th, imgSrc = cv.threshold(imgSrc, 23, 255, cv.THRESH_BINARY)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    imgSkeletons = Skeletons(imgSrc, kernel)

    plt.subplot(131), plt.imshow(imgSrc, cmap="gray")
    plt.title(u"Source", fontproperties=font_set)

    plt.subplot(132), plt.imshow(imgSkeletons, cmap="gray")
    plt.title(u"Skeletons", fontproperties=font_set)

    imgConvexHull = ConvexHull(imgSrc)
    imgConvexHull = cv.medianBlur(imgConvexHull, 9)
    imgOut = Dilate(imgConvexHull - imgSrc, kernel)

    plt.subplot(133), plt.imshow(imgOut, cmap="gray")
    plt.title(u"ConvexHull", fontproperties=font_set)

    plt.show()

    '''
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/Calculator/spot2.jpg")
    imgSrc = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    imgSrc = Not(imgSrc)

    th, imgSrc = cv.threshold(imgSrc,125, 255, cv.THRESH_BINARY)
    imgDstThining = Thining(imgSrc)

    plt.figure(0)
    plt.subplot(231), plt.imshow(imgSrc, cmap="gray")
    plt.title(u"Source", fontproperties=font_set)
    plt.subplot(232), plt.imshow(imgDstThining, cmap="gray")
    plt.title(u"Thining", fontproperties=font_set)

    plt.subplot(233), plt.imshow(imgSrc - imgDstThining, cmap="gray")
    plt.title(u"Source - Thicking", fontproperties=font_set)

    imgDstThicking = Thicking(imgSrc)
    plt.subplot(234), plt.imshow(imgDstThicking, cmap="gray")
    plt.title(u"Thicking", fontproperties=font_set)

    plt.subplot(235), plt.imshow(imgDstThicking - imgSrc, cmap="gray")
    plt.title(u"Thicking - Source", fontproperties=font_set)

    plt.subplot(236), plt.imshow(imgDstThicking - imgDstThining, cmap="gray")
    plt.title(u"Thicking - Source", fontproperties=font_set)

    plt.show()
    '''
    '''
    floodFilled = np.zeros((imgSrc.shape[0] + 2, imgSrc.shape[1] + 2), dtype=np.uint8)
    imgDst = imgSrc.copy()
    cv.floodFill(imgDst, floodFilled, (0, 0), 255)
    imgDst = imgSrc + Not(imgDst)
    
    plt.subplot(122), plt.imshow(imgDst, cmap="gray")
    plt.title(u"FillHole", fontproperties=font_set)

    hit = np.ones([16,16], dtype=np.uint8)
    miss = np.ones([18,18], dtype=np.uint8)
    miss[int(miss.shape[0] / 2 - hit.shape[0] / 2):int(miss.shape[0] / 2 + hit.shape[0] / 2),
         int(miss.shape[1] / 2 - hit.shape[1] / 2):int(miss.shape[1] / 2 + hit.shape[1] / 2)] = 0
    kernel = np.array(-miss, dtype=np.int8)
    kernel[int(miss.shape[0] / 2 - hit.shape[0] / 2):int(miss.shape[0] / 2 + hit.shape[0] / 2),
         int(miss.shape[1] / 2 - hit.shape[1] / 2):int(miss.shape[1] / 2 + hit.shape[1] / 2)] = 1

    imgDst = MyHitMiss(imgSrc, hit, miss)
    plt.figure(1), plt.imshow(imgDst, cmap="gray")
    plt.title(u"HitMiss1", fontproperties=font_set)

    imgDst = cv.morphologyEx(imgSrc, cv.MORPH_HITMISS, kernel)
    plt.figure(2), plt.imshow(imgDst, cmap="gray")
    plt.title(u"HitMiss2", fontproperties=font_set)
    '''

    '''
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/Calculator/spot.jpg")
    kernel = np.empty([32,32], dtype=np.uint8)
    cv.circle(kernel, (16,16), 16, 255)
    imgOut = cv.morphologyEx(imgSrc, cv.MORPH_CLOSE, kernel)

    plt.figure(0)
    plt.subplot(231), plt.imshow(imgSrc)
    plt.title(u"原图", fontproperties=font_set)

    plt.subplot(232), plt.imshow(imgOut)
    plt.title(u"闭操作", fontproperties=font_set)

    kernel = np.empty([105, 105], dtype=np.uint8)
    cv.circle(kernel, (52, 52), 7, 255)
    imgOut = cv.morphologyEx(imgOut, cv.MORPH_OPEN, kernel)

    kernel = np.ones([120,120],dtype=np.uint8)
    imgOut = cv.morphologyEx(imgOut, cv.MORPH_OPEN, kernel)

    plt.subplot(233), plt.imshow(imgOut)
    plt.title(u"开操作", fontproperties=font_set)

    kernel = np.ones([3, 3], dtype=np.uint8)
    imgOut = cv.morphologyEx(imgOut, cv.MORPH_GRADIENT, kernel)
    plt.subplot(234), plt.imshow(imgOut)
    plt.title(u"形态学梯度操作", fontproperties=font_set)
    '''
    '''
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/Calculator/caculator.jpg")
    kernel = np.ones([3, 20], dtype=np.uint8)
    imgOut = RebuildMorphOpen(imgSrc, kernel, 2)
    
    # 因为要要过滤掉符号，而计算器上的符号的要比键盘的灰度级要大（每个通道，因为符号是白色的）
    # 所以使用开操作，顶帽操作以后，就可以把背景过滤掉，从而提取出符号
    plt.figure(0)
    plt.subplot(231), plt.imshow(imgSrc)
    plt.title(u"原图", fontproperties=font_set)

    plt.subplot(232), plt.imshow(imgOut)
    plt.title(u"重建开操作1", fontproperties=font_set)

    imgSrc2 = imgSrc - imgOut
    plt.subplot(233), plt.imshow(imgSrc2)
    plt.title(u"重建顶帽操作1 (水平线删除)", fontproperties=font_set)

    kernel = np.ones([22, 1], dtype=np.uint8)
    imgOut = RebuildMorphOpen(imgSrc2, kernel, 1)

    plt.subplot(234), plt.imshow(imgOut)
    plt.title(u"重建开操作2", fontproperties=font_set)

    imgSrc3 = imgSrc2 - imgOut
    plt.subplot(235), plt.imshow(imgSrc3)
    plt.title(u"重建顶帽操作2（垂直线删除）", fontproperties=font_set)

    imgSrc3 = cv.cvtColor(imgSrc3, cv.COLOR_RGB2GRAY)
    th, imgOut = cv.threshold(imgSrc3, 70, 255, cv.THRESH_BINARY)

    kernel = np.ones([2,2],dtype=np.uint8)
    imgOut = Dilate(imgOut, kernel)

    plt.subplot(236), plt.imshow(imgOut, cmap="gray")
    plt.title(u"二值化", fontproperties=font_set)

    plt.show()
    '''














