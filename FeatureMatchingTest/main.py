import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from feature import *
from contours import *
from siftInner import *

def DrawPoint(imgSrc:np.ndarray, imgCorner:np.ndarray):
    points = np.where(imgCorner == 0xff)
    for i, j in zip(points[1], points[0]):
        cv.circle(imgSrc, (i, j) , 3, (0xff, 0, 0xff), -1)


if __name__ == "__main__":
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")

    factory = SiftFeature2D(imgSrc)
    factory.GetFeatures()

    '''
    imgDst = PCAImage(imgSrc)
    # imgDst = PCAImage(cv.transpose(imgSrc))

    plt.figure(1)
    plt.subplot(121), plt.imshow(imgSrc), plt.title("Source", fontproperties=font_set)
    plt.subplot(122), plt.imshow(imgDst), plt.title("Result", fontproperties=font_set)
    plt.show()
    '''


    '''
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/man.png")
    # imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    imgGray = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)
    cv.normalize(imgGray, imgGray, 0, 255, cv.NORM_MINMAX)
    # th, imgGray = cv.threshold(imgGray, 23, 255, cv.THRESH_BINARY)
    imgGray = cv.threshold(np.uint8(imgGray), 0, 255, cv.THRESH_OTSU)[1]
    imgDst = Moore(imgGray)

    plt.figure(1)
    plt.subplot(121), plt.imshow(imgGray, cmap="gray"), plt.title("Source", fontproperties=font_set)
    plt.subplot(122), plt.imshow(imgDst, cmap="gray"), plt.title("Result", fontproperties=font_set)
    plt.show()
    '''
    '''
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    PCAImage(imgSrc)
    # imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    imgGray = cv.cvtColor(imgSrc, cv.COLOR_RGB2GRAY)

    cv.normalize(imgGray, imgGray, 0, 255, cv.NORM_MINMAX)
    #th, imgGray = cv.threshold(imgGray, 23, 255, cv.THRESH_BINARY)
    imgGray = cv.threshold(np.uint8(imgGray), 0, 255, cv.THRESH_OTSU)[1]
    imgDst = Skeletons(imgGray)
    DrawSkeletonsKeyPoints(imgDst, 8, 15, 29)

    plt.figure(1)
    plt.subplot(121), plt.imshow(imgGray, cmap="gray"), plt.title("Source", fontproperties=font_set)
    plt.subplot(122), plt.imshow(imgDst,  cmap="gray"), plt.title("Result", fontproperties=font_set)
    plt.show()
    '''

    '''
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    imgSrc1, imgDst = EigImage(imgSrc)

    plt.figure(1), plt.imshow(imgSrc1, cmap="gray"), plt.title("Source", fontproperties=font_set)
    plt.figure(2), plt.imshow(imgDst, cmap="gray"), plt.title("Result", fontproperties=font_set)
    # plt.show()

    # ע����������ֵ���Ƶķ���
    # �Լ�Harris�ǵ���ķ���
    imgDst = CornerHarris(imgSrc, 0.01)
    maxM = np.max(imgDst)
    imgDilate = cv.dilate(imgDst, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    imgLocalMax = cv.compare(imgDilate, imgDst, cv.CMP_EQ)
    imgCorner = cv.threshold(imgDst, 0.01 * maxM, 0xff, cv.THRESH_BINARY)[1]
    imgCorner = cv.bitwise_and(imgCorner, np.float32(imgLocalMax))
    imgResult = imgSrc.copy()
    
    DrawPoint(imgResult, imgCorner)

    # opencv��׼Harris�ǵ��ⷽ��
    imgDst = cv.cornerHarris(cv.cvtColor(np.float32(imgSrc), cv.COLOR_RGB2GRAY), 3, 3, 0.01)
    maxM = np.max(imgDst)
    imgDilate = cv.dilate(imgDst, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    imgLocalMax = cv.compare(imgDilate, imgDst, cv.CMP_EQ)
    imgCorner = cv.threshold(imgDst, 0.01 * maxM, 0xff, cv.THRESH_BINARY)[1]
    imgCorner = cv.bitwise_and(imgCorner, np.float32(imgLocalMax))
    imgResult2 = imgSrc.copy()
    DrawPoint(imgResult2, imgCorner)

    plt.figure(3), plt.imshow(imgResult), plt.title("Result1", fontproperties=font_set)
    plt.figure(4), plt.imshow(imgResult2), plt.title("Result2", fontproperties=font_set)
    plt.show()
    '''

