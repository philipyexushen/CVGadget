import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from feature import *
from contours import *
from siftInner import *

def DrawPoint(imgSrc:np.ndarray, imgCorner:np.ndarray):
    points = np.where(imgCorner == 0xff)
    for i, j in zip(points[1], points[0]):
        cv.circle(imgSrc, (i, j) , 3, (0xff, 0, 0xff), -1)

def DrawFeature(imgSrc:np.ndarray, feature_array, index_choice):
    imgDst = imgSrc.copy()
    for i, feature in enumerate(feature_array):
        if i not in index_choice:
            continue
        point = feature[0]
        h, w, _, octave = point[:4]

        real_w = int(round(w * (2 ** (octave - 1))))
        real_h = int(round(h * (2 ** (octave - 1))))
        cv.circle(imgDst, (real_w, real_h), 5, (255, 255, 0))

        angle = feature[1]
        real_h_pt = int(real_h + 10 * np.sin(angle))
        real_w_pt = int(real_w + 10 * np.cos(angle))

        cv.arrowedLine(imgDst, (real_w, real_h), (real_w_pt, real_h_pt), (255, 255, 0))
    return imgDst

if __name__ == "__main__":
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    # imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    imgSrc = plt.imread("F:/win10/Philip/Documents/MATLAB/sift/scene.pgm")
    imgSrc2 = plt.imread("F:/win10/Philip/Documents/MATLAB/sift/book.pgm")
    # imgSrc = plt.imread("E:/Users/Administrator/pictures/queen/13.jpg")
    factory1 = SiftFeature2D(imgSrc)
    descriptor1 = factory1.GetFeatures()
    factory2 = SiftFeature2D(imgSrc2)
    descriptor2 = factory2.GetFeatures()
    '''
    feature1_index, feature2_index = Match(descriptor1, descriptor2)

    print(feature1_index)
    print(feature2_index)
    imgDst1 = DrawFeature(factory1.image_source, factory1.feature_array, feature1_index)
    imgDst2 = DrawFeature(factory2.image_source, factory2.feature_array, feature2_index)

    plt.figure(1)
    plt.subplot(121)
    if len(np.shape(factory1.image_source)) == 3:
        imgDst = cv.cvtColor(imgDst1, cv.COLOR_BGR2RGB)
        plt.imshow(imgDst)
        plt.show()
    elif len(np.shape(factory1.image_source)) == 2:
        plt.imshow(imgDst1, cmap="gray")

    plt.subplot(122)
    if len(np.shape(factory2.image_source)) == 3:
        imgDst = cv.cvtColor(imgDst1, cv.COLOR_BGR2RGB)
        plt.imshow(imgDst2)
    elif len(np.shape(factory2.image_source)) == 2:
        plt.imshow(imgDst2, cmap="gray")

    plt.show()
    '''
    matched = Match2(descriptor1, descriptor2)


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

