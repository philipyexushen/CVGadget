# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from WarpTransform import *
from multiprocessing.dummy import Process

_windowCount = int(0)
_mainWinName = "source"

def WarpImage(imgSrc:np.ndarray, dots:tuple, count)->None:
    imgBLiner, imgNearest = WarpCorrection(imgSrc, dots)

    winName:str = f"result BLiner {count}"
    cv.namedWindow(winName)
    cv.imshow(winName, imgBLiner)

    winName:str = f"result nearest {count}"
    cv.namedWindow(winName)
    cv.imshow(winName, imgNearest)

    cv.waitKey(0)
    cv.destroyWindow(winName)


class WarpCorrectionMgr:
    def __init__(self, imgSrc):
        self.__clickTime = 0
        self.__imgSrc = imgSrc.copy()
        self.__imgDrawn = imgSrc.copy()
        self.__dots = []

    @property
    def sourceImage(self):
        return self.__imgSrc

    @property
    def drawnImage(self):
        return self.__imgDrawn

    @drawnImage.setter
    def drawnImage(self, newImg):
        self.__imgDrawn = newImg

    @property
    def clickTime(self):
        return self.__clickTime

    @clickTime.setter
    def clickTime(self, v):
        self.__clickTime = v

    @property
    def dots(self):
        return self.__dots

    @staticmethod
    def MouseCallback(event, x, y, flags, param):
        # 四个点的顺序一定要按照左上，右上，右下，左下的顺时针顺序点
        if event == cv.EVENT_LBUTTONDBLCLK:
            clickTime = param.clickTime
            cv.circle(param.drawnImage, (x, y), 8, (0, 0, 255),-1)
            param.dots.append((x, y))
            cv.imshow(_mainWinName, param.drawnImage)

            if clickTime + 1 == 4:
                global _windowCount
                p = Process(target=WarpImage, args=(param.sourceImage, param.dots.copy(), _windowCount))
                p.daemon = True
                p.start()

                param.drawnImage = param.sourceImage.copy()
                cv.imshow(_mainWinName, param.sourceImage)
                param.dots.clear()
                _windowCount += 1

            param.clickTime = (clickTime + 1) % 4


if __name__ == "__main__":
    font_set = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=12)
    '''
    imgSrc = plt.imread("E:/Users/Administrator/pictures/Test/user.jpg")
    op = np.transpose(MoveT(10,30, RotateT(np.pi/12, ZoomT(1.1, 1.2))))

    imgDst = Transform(imgSrc, op)
    plt.figure(1), plt.imshow(imgDst), plt.title("Result", fontproperties=font_set)
    plt.show()
    '''
    cv.namedWindow(_mainWinName)
    imgSrc = cv.imread("E:/Users/Administrator/pictures/Test/skew.jpg")
    imgSrc = cv.resize(imgSrc, (int(imgSrc.shape[1]/4), int(imgSrc.shape[0]/4)))

    mgr = WarpCorrectionMgr(imgSrc)
    cv.setMouseCallback(_mainWinName, WarpCorrectionMgr.MouseCallback, mgr)

    cv.imshow(_mainWinName, imgSrc)
    cv.waitKey(0)
    cv.destroyAllWindows()


