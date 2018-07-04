from common import *
from matplotlib import pyplot as plt
from numba import jit

class SiftFeature2D:
    def __init__(self, imgSrc:np.ndarray):
        self.__sift_img_border = 1
        self.__extreme_point_value_threshold = 0.04
        self.__imgSrc:np.ndarray = imgSrc

    def __BuildDogPyramid(self, imgSrc, init_sigma, octaves_layer_num ,octaves_num):
        k = 2**(1 / octaves_layer_num)
        sigma = np.zeros([octaves_layer_num + 3], dtype=np.float64)
        sigma[0] = init_sigma

        for i in range(1, octaves_layer_num + 3):
            prev_sigma = init_sigma*(k**(i - 1))
            cur_sigma = prev_sigma*k
            sigma[i] = np.sqrt(cur_sigma**2 - prev_sigma**2)

        pyramid_list = []
        cur_height, cur_width = self.__imgSrc.shape[:2]
        # 先构造高斯金字塔
        for octave in range(octaves_num):
            pyramid_list.append(np.zeros([octaves_layer_num + 3, cur_height, cur_width], dtype=np.uint8))
            current_layer_pack = pyramid_list[octave]
            for layer in range(octaves_layer_num + 3):
                if octave == 0 and layer == 0:
                    # 第0层第0张图片，当然是原图了
                    current_layer_pack[layer] = imgSrc
                elif layer == 0:
                    # 每一组第0副图像时上一组倒数第三幅图像隔点采样得到（尺度刚好是2 sigma）
                    current_layer_pack[layer] = cv.resize(pyramid_list[octave - 1][octaves_layer_num],
                                                          dsize=(cur_width,cur_height ), interpolation=cv.INTER_NEAREST)
                else:
                    prev_img = current_layer_pack[layer - 1]
                    current_layer_pack[layer] \
                        = cv.GaussianBlur(prev_img, (3, 3), sigmaX=sigma[layer], sigmaY=sigma[layer])

            cur_height = int(np.round(cur_height / 2))
            cur_width = int(np.round(cur_width / 2))

        # 构造DOG金字塔，由高斯金字塔两两相减
        cur_height, cur_width = self.__imgSrc.shape[:2]
        pyramid_DOG_list = []
        for octave in range(octaves_num):
            pyramid_DOG_list.append(np.zeros([octaves_layer_num + 3, cur_height, cur_width], dtype=np.uint8))

            current_layer_pack = pyramid_list[octave]
            current_DOG_layer_pack = pyramid_DOG_list[octave]
            for layer in range(octaves_layer_num + 2):
                first_mat = current_layer_pack[layer]
                second_mat = current_layer_pack[layer + 1]
                current_DOG_layer_pack[layer] = np.subtract(second_mat, first_mat)

            cur_height = int(np.round(cur_height / 2))
            cur_width = int(np.round(cur_width / 2))

        return pyramid_DOG_list

    @jit
    def __isMaximumPoint(self, img_current, img_prev, img_next, r, c, threasold):
        val = img_current[r, c]
        if np.abs(val) <= threasold:
            return False

        is_extreme_val_positive = True \
            if  val > 0 \
                and val > img_prev[r - 1, c - 1] and val > img_prev[r, c - 1] and val > img_prev[r + 1, c - 1]\
                and val > img_prev[r, c] and val > img_prev[r, c] and val > img_prev[r, c + 1]\
                and val > img_prev[r + 1, c - 1] and val > img_prev[r + 1, c] and val > img_prev[r + 1, c + 1] \
                and val > img_next[r - 1, c - 1] and val > img_next[r, c - 1] and val > img_next[r + 1, c - 1] \
                and val > img_next[r, c] and val > img_next[r, c] and val > img_next[r, c + 1] \
                and val > img_next[r + 1, c - 1] and val > img_next[r + 1, c] and val > img_next[r + 1, c + 1] else False

        is_extreme_val_negative = True \
            if  val < 0 \
                and val < img_prev[r - 1, c - 1] and val < img_prev[r, c - 1] and val < img_prev[r + 1, c - 1]\
                and val < img_prev[r, c] and val < img_prev[r, c] and val < img_prev[r, c + 1]\
                and val < img_prev[r + 1, c - 1] and val < img_prev[r + 1, c] and val < img_prev[r + 1, c + 1] \
                and val < img_next[r - 1, c - 1] and val < img_next[r, c - 1] and val < img_next[r + 1, c - 1] \
                and val < img_next[r, c] and val < img_next[r, c] and val < img_next[r, c + 1] \
                and val < img_next[r + 1, c - 1] and val < img_next[r + 1, c] and val < img_next[r + 1, c + 1] else False

        return is_extreme_val_positive or is_extreme_val_negative

    

    @MethodInformProvider
    def __AccurateKeyPointLocalization(self, imgSrc, pyramid_DOG_list, octaves_layer_num, octaves_num):
        w, h = imgSrc.shape[:2]
        border = self.__sift_img_border
        threshold = self.__extreme_point_value_threshold
        for octave in range(octaves_num):
            for layer in range(1, octaves_layer_num - 1):
                img_prev = pyramid_DOG_list[octave][layer - 1]
                img_current = pyramid_DOG_list[octave][layer]
                img_next = pyramid_DOG_list[octave][layer + 1]

                for r, c in zip(range(border, h - border), range(border, w - border)):
                    # 是否是极值点
                    if not self.__isMaximumPoint(img_current, img_prev, img_next, r, c, threshold):
                        continue



    @MethodInformProvider
    def GetFeatures(self, init_sigma = 0.5, octaves_layer_num = 5, octaves_num = 3):
        imgSrc = self.__imgSrc
        if np.shape(imgSrc)[2] == 3:
            imgSrc = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)

        pyramid_DOG_list = self.__BuildDogPyramid(imgSrc, init_sigma, octaves_layer_num, octaves_num)

        self.__AccurateKeyPointLocalization(imgSrc, pyramid_DOG_list, octaves_layer_num, octaves_num)

        '''
                i = 0
        for octave in range(octaves_num):
            for layer in range(octaves_layer_num + 2):
                plt.figure(i)
                plt.imshow(pyramid_DOG_list[octave][layer], cmap="gray")
                i += 1
        plt.show()
        '''



