from Common import *


def MoveT(tx, ty, lastOp = None)->np.ndarray:
    op = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    if lastOp is not None:
        op = np.dot(op, lastOp)
    return op


def RotateT(r, lastOp = None)->np.ndarray:
    op = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    if lastOp is not None:
        op = np.dot(op, lastOp)
    return op


def ZoomT(rx, ry, lastOp = None)->np.ndarray:
    op = np.array([[rx, 0, 0], [0, ry, 0], [0, 0, 1]], dtype=np.float32)
    if lastOp is not None:
        op = np.dot(op, lastOp)
    return op


@MethodInformProvider
def Transform(imgSrc:np.ndarray, op)->np.ndarray:
    h, w = imgSrc.shape[:2]
    imgDst = np.zeros(imgSrc.shape, dtype=imgSrc.dtype)
    for i in range(h):
        v1 = np.stack((np.arange(w), np.ones(w)*i, np.ones(w)),axis=-1)
        v2 = np.dot(v1, op)
        tpx, tpy, tpz = np.hsplit(v2, 3)
        for iy, ix, iz, j in zip(tpy, tpx, tpz, range(w)):
            py, px = int(iy/iz), int(ix/iz)
            if 0<= py < h and 0 <= px < w:
                imgDst[int(py), int(px)] = imgSrc[i, j]

    return imgDst

@jit
def BilinearInterpolation(imgSrc:np.ndarray, h, w, sx:float, sy:float)->float:
    """
    对图片的指定位置做双线性插值
    :param imgSrc:源图像
    :param h: src的高度
    :param w: src的宽度
    :param sx: x位置
    :param sy: y位置
    :return: 所插入的值
    """
    intSx, intSy = int(sx), int(sy)
    if 0 <= intSx  < w - 1 and 0 <= intSy < h - 1:
        x1, x2 = intSx, intSx + 1
        y1, y2 = intSy, intSy + 1
        H1 = np.dot(np.array([x2 - sx, sx - x1]), imgSrc[y1: y2 + 1, x1:x2 + 1])
        return H1[0]*(y2 - sy) + H1[1]*(sy - y1)
    else:
        return imgSrc[intSy, intSx]


@MethodInformProvider
def WarpCorrection(imgSrc:np.ndarray, dots:tuple)->(np.ndarray,np.ndarray):
    assert len(dots) == 4

    # 四个点的顺序一定要按照左上，右上，右下，左下的顺时针顺序点
    d1, d2, d3, d4 = dots
    x1, x2, x3, x4 = d1[0], d2[0], d3[0], d4[0]
    y1, y2, y3, y4 = d1[1], d2[1], d3[1], d4[1]
    assert x1 < x2
    assert x4 < x3
    assert y1 < y4
    assert y2 < y3

    objW = np.round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    objH = np.round(np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2))

    # 在这里我简单地设为把所输入的四个点的位置，通过2D变换，变换为长方形的四个顶点的位置（以x1为起点）
    t1, t2, t3, t4 = (y1, x1), (y1, x1 + objW), (y1 + objH, x1 + objW), (y1 + objH, x1),

    rx1, rx2, rx3, rx4 = t1[1], t2[1], t3[1], t4[1]
    ry1, ry2, ry3, ry4 = t1[0], t2[0], t3[0], t4[0]

    # ================Step 0: 根据 8个点两两对应关系找到Homography矩阵================
    # 把8个约束写成方程组，以矩阵的形式表达
    m = np.array([
                  [y1, x1, 1, 0, 0, 0, -ry1 * y1, -ry1 * x1],
                  [0, 0, 0, y1, x1, 1, -rx1 * y1, -rx1 * x1],
                  [y2, x2, 1, 0, 0, 0, -ry2 * y2, -ry2 * x2],
                  [0, 0, 0, y2, x2, 1, -rx2 * y2, -rx2 * x2],
                  [y3, x3, 1, 0, 0, 0, -ry3 * y3, -ry3 * x3],
                  [0, 0, 0, y3, x3, 1, -rx3 * y3, -rx3 * x3],
                  [y4, x4, 1, 0, 0, 0, -ry4 * y4, -ry4 * x4],
                  [0, 0, 0, y4, x4, 1, -rx4 * y4, -rx4 * x4],
                ])

    vectorSrc = np.array([ry1, rx1, ry2, rx2, ry3, rx3, ry4, rx4])
    vectorSrc.shape = (1, 8)
    HFlat = np.dot(np.linalg.inv(m), np.transpose(vectorSrc))
    a, b, c, d, e, f, g, h = HFlat[0, 0],HFlat[1, 0],HFlat[2, 0],HFlat[3, 0],HFlat[4, 0],HFlat[5, 0],HFlat[6, 0],HFlat[7, 0]

    H = np.array([[a, b, c],
                  [d, e, f],
                  [g, h, 1]], dtype=np.float32)

    # ================Step 1: 通过对原图像四个顶点进行正向投射变换，确定目标图像区域================
    height, width = imgSrc.shape[:2]
    matrixOriginVertex = np.array([[0, 0, 1],
                                   [0, width - 1, 1],
                                   [height - 1, width - 1, 1] ,
                                   [height - 1, 0, 1]])

    result = np.dot(matrixOriginVertex, np.transpose(H))
    minX = int(min(result[0, 1]/result[0, 2], result[1, 1]/result[1, 2], result[2, 1]/result[2, 2], result[3, 1]/result[3, 2]))
    maxX = int(max(result[0, 1]/result[0, 2], result[1, 1]/result[1, 2], result[2, 1]/result[2, 2], result[3, 1]/result[3, 2]))
    minY = int(min(result[0, 0]/result[0, 2], result[1, 0]/result[1, 2], result[2, 0]/result[2, 2], result[3, 0]/result[3, 2]))
    maxY = int(max(result[0, 0]/result[0, 2], result[1, 0]/result[1, 2], result[2, 0]/result[2, 2], result[3, 0]/result[3, 2]))

    # ================Step 2: 反向变换+双二次插值校正图像================
    vtr = np.empty((0,3),dtype=np.float32)
    for i in range(minY, maxY):
        arr1 = np.arange(minX, maxX)
        arr2 = np.ones(maxX - minX)
        vt1 = np.stack((arr2*i, arr1 , arr2), axis=-1)
        vtr = np.concatenate((vtr, vt1), axis=0)

    # 请注意，因为传进去的是规范化后(Y, X, 1)的值，所以得到的其实是(y/Z, x/Z, 1/Z的值)
    vts = np.dot(vtr,np.linalg.inv(np.transpose(H)))
    dstHeight, dstWidth = maxY - minY + 1, maxX - minX + 1
    imgBLiner = np.zeros((dstHeight, dstWidth, imgSrc.shape[2]), dtype=imgSrc.dtype)
    imgNearest = np.zeros((dstHeight, dstWidth, imgSrc.shape[2]), dtype=imgSrc.dtype)
    
    for (r, s) in zip(vtr, vts):
        ry, rx = int(r[0]), int(r[1])
        iy, ix = s[:2]
        # 需要解 [y, x] = [iy*(g*y + h*x + 1), ix*(g*y + h*x + 1)]这个方程
        TH = np.linalg.inv(np.array([[iy * g - 1, iy * h],
                                     [ix * g, ix * h - 1]]))

        vxy = np.dot(TH, np.array([[-iy], [-ix]]))
        sy, sx = vxy[0, 0], vxy[1, 0]

        if 0 <= round(sy) < height and 0 <= round(sx) < width:
            imgBLiner[ry - minY, rx - minX] = BilinearInterpolation(imgSrc, height, width, sx, sy)
            imgNearest[ry - minY, rx - minX] = imgSrc[int(round(sy)),int(round(sx))]

    return imgBLiner, imgNearest




    
    













