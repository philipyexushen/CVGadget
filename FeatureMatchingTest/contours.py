from common import *

@MethodInformProvider
def Moore(imgSrc:np.ndarray)->np.ndarray:
    h, w = imgSrc.shape[:2]

    @jit
    def _NextDirect(b, c):
        diff = (b[0] - c[0], b[1] - c[1])
        if diff[0] == 0 and diff[1] == 1:# p8 -> p9
            return c[0] - 1, c[1]
        elif diff[0] == 1 and diff[1] == 1: # p9 -> p2
            return c[0], c[1] + 1
        elif diff[0] == 1 and diff[1] == 0: # p2 -> p3
            return c[0], c[1] + 1
        elif diff[0] == 1 and diff[1] == -1: # p3 -> p4
            return c[0] + 1, c[1]
        elif diff[0] == 0 and diff[1] == -1: # p4 -> p6
            return c[0] + 1, c[1]
        elif diff[0] == -1 and diff[1] == -1: # p5 -> p6
            return c[0], c[1] - 1
        elif diff[0] == -1 and diff[1] == 0: # p6 -> p7
            return c[0], c[1] - 1
        elif diff[0] == -1 and diff[1] == 1: # p7 -> p8
            return c[0] - 1, c[1]

    def _InnerScan(imgSrc:np.ndarray, imgDst:np.ndarray, imgFlag:np.ndarray, h, w, b0):
        c1 = (b0[0] - 1, b0[1])
        b1 = b0
        while True:
            bFound = bool(False)
            lastPos = c1
            for i in range(8):
                curPos = _NextDirect(b1, lastPos)
                if imgSrc[curPos] == 255:
                    b1 = curPos
                    imgDst[b1] = 255
                    c1 = lastPos
                    bFound = True
                    break
                lastPos = curPos

            if bFound and b1 == b0:
                break


    imgDst = np.zeros(imgSrc.shape, dtype=imgSrc.dtype)
    imgFlag = np.zeros(imgSrc.shape, dtype=bool)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if imgSrc[i, j] == 255 and imgFlag[i, j] == 0:
                b0 = (i, j)
                _InnerScan(imgSrc, imgDst, imgFlag, h, w, b0)

                # 懒得找多个了，先放这里把
                return imgDst

    return imgDst


@jit
def _FilterOver(imgSrc:np.ndarray,imgFlag, h, w):
    points = np.where(imgSrc == 255)
    for py, px in zip(points[0], points[1]):
        if 1 <= py < h - 1 and 1 <= px < w - 1:
            listPAdjoin = (py - 1, px), (py - 1, px + 1), (py, px + 1), \
                          (py + 1, px + 1), (py + 1, px), (py + 1, px - 1), \
                          (py, px - 1), (py - 1, px - 1)

            if listPAdjoin[0] == 255 or listPAdjoin[1] == 255 or listPAdjoin[7] == 255 or listPAdjoin[6] == 255:
                imgFlag[py, px] = True

    points = np.where(imgFlag == True)
    if points and len(points[0]) != 0:
        # print("step %d last: %d" % (method, len(points[0])))
        for py, px in zip(points[0], points[1]):
            imgSrc[py, px] = 0


@jit
def _MarkPoint(imgSrc, imgFlag, h, w, method:int)->bool:
    points = np.where(imgSrc == 255)
    for py, px in zip(points[0], points[1]):
        if 1 <= py < h - 1 and 1 <= px < w - 1:
            listPAdjoin = (py - 1, px), (py - 1, px + 1), (py, px + 1),\
                                (py + 1, px + 1), (py + 1, px), (py + 1, px - 1), \
                                (py, px - 1), (py - 1, px - 1)
            countN = 0
            countT = 0
            for i in range(8):
                if imgSrc[listPAdjoin[i]] == 255:
                    countN += 1
                # imgSrc[listPAdjoin[i]] == 0:
                elif imgSrc[listPAdjoin[(i + 1) % 8]] == 255:
                    countT += 1

            assert method == 1 or method == 2
            if method == 1:
                if 2 <= countN <= 6 \
                    and countT == 1 \
                    and (imgSrc[listPAdjoin[0]] == 0 or imgSrc[listPAdjoin[2]] == 0 or imgSrc[listPAdjoin[4]] == 0) \
                    and (imgSrc[listPAdjoin[2]] == 0 or imgSrc[listPAdjoin[4]] == 0 or imgSrc[listPAdjoin[6]] == 0):
                    imgFlag[py, px] = True
            elif method == 2:
                if 2 <= countN <= 6 \
                    and countT == 1 \
                    and (imgSrc[listPAdjoin[0]] == 0 or imgSrc[listPAdjoin[2]] == 0 or imgSrc[listPAdjoin[6]] == 0) \
                    and (imgSrc[listPAdjoin[0]] == 0 or imgSrc[listPAdjoin[4]] == 0 or imgSrc[listPAdjoin[6]] == 0):
                    imgFlag[py, px] = True

    points = np.where(imgFlag == True)
    if points and len(points[0]) != 0:
        # print("step %d last: %d" % (method, len(points[0])))
        for py, px in zip(points[0], points[1]):
            imgSrc[py, px] = 0
        return True
    return False


@MethodInformProvider
def DrawSkeletonsKeyPoints(imgSrc:np.ndarray, kernelSize, minThresh, maxThresh):
    @jit
    def _Scan(imgNorm:np.ndarray, kernelSize, minThresh, maxThresh):
        h, w = imgNorm.shape[:2]
        points = []
        for i in range(h):
            for j in range(w):
                val = imgNorm[i, j]
                if val != 1:
                    continue

                count = 0
                for y in range(-kernelSize, kernelSize):
                    for x in range(-kernelSize, kernelSize):
                        if y == 0 and x == 0:
                            continue
                        py, px = i + y, j + x
                        count += imgNorm[py, px]

                if 1 <= count < minThresh or count > maxThresh:
                    points.append((i, j))

        return points

    imgNorm = np.empty(imgSrc.shape, dtype=imgSrc.dtype)
    cv.normalize(imgSrc, imgNorm, 0, 1, cv.NORM_MINMAX)

    points = _Scan(imgNorm,kernelSize, minThresh, maxThresh)
    for py, px in points:
        cv.circle(imgSrc, (px, py), 4, 255)


@MethodInformProvider
def Skeletons(imgSrc:np.ndarray, maxStep:int = -1)->np.ndarray:
    assert len(imgSrc.shape) == 2

    imgDst = imgSrc.copy()
    imgFlag = np.empty(imgSrc.shape, dtype=bool)

    h, w = imgSrc.shape[:2]
    stepCount = 0
    while True:
        if maxStep != -1 and stepCount > maxStep:
            break
        imgFlag.fill(False)
        bHasFlag = _MarkPoint(imgDst, imgFlag, h, w, 1)
        if not bHasFlag:
            break

        imgFlag.fill(False)
        bHasFlag = _MarkPoint(imgDst, imgFlag, h, w, 2)
        if not bHasFlag:
            break

    imgFlag.fill(False)
    _FilterOver(imgDst, imgFlag, h, w)

    return imgDst