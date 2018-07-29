from common import *
from matplotlib import pyplot as plt
from numba import jit

class SiftFeature2D:
    def __init__(self, imgSrc:np.ndarray):
        self.__extreme_point_value_threshold = 0.04
        self.__imgSrc:np.ndarray = imgSrc
        self.__sift_img_border = 5
        self.__sift_contour_threshold = 10.0

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
            pyramid_DOG_list.append(np.zeros([octaves_layer_num + 3, cur_height, cur_width], dtype=np.int16))

            current_layer_pack = pyramid_list[octave]
            current_DOG_layer_pack = pyramid_DOG_list[octave]
            for layer in range(octaves_layer_num + 2):
                first_mat = np.int16(current_layer_pack[layer])
                second_mat = np.int16(current_layer_pack[layer + 1])
                current_DOG_layer_pack[layer] = np.subtract(second_mat, first_mat)

            cur_height = int(np.round(cur_height / 2))
            cur_width = int(np.round(cur_width / 2))

        return pyramid_DOG_list

    @staticmethod
    def __isMaximumPoint(pyramid_DOG_list ,octave, layer, r, c, threshold):
        img_prev = pyramid_DOG_list[octave][layer - 1]
        img_current = pyramid_DOG_list[octave][layer]
        img_next = pyramid_DOG_list[octave][layer + 1]
        val = img_current[r, c]

        # 小于某个阈值，直接扔掉
        if np.abs(val) <= threshold:
            return False

        value = img_current[r, c]
        img_block_current = img_current[r - 1:r + 2, c - 1:c + 2]
        img_block_prev = img_prev[r - 1:r + 2, c - 1:c + 2]
        img_block_next = img_next[r - 1:r + 2, c - 1:c + 2]

        if value > 0 and value >= np.max(img_block_current) and value >= np.max(img_block_prev) and value >= np.max(img_block_next):
            return True

        return value <= np.min(img_block_current) and value <= np.min(img_block_prev) and value <= np.min(img_block_next)

    def __adjustLocalExtrema(self, pyramid_DOG_list, octave, vector, octaves_layer_num, key_point_array: list):
        """
        插值和删除边缘效应
        :return:
        """
        current_pyramid_DOG = pyramid_DOG_list[octave]
        r, c, layer = vector
        xr, xc, xs = (0.0, 0.0, 0.0)
        # 注意我们现在要用三维的眼光去看到插值了，因为我们要考虑当前的层以及前后两层
        img_scale = 1.0 / 255.0
        # 插值最大次数，防止不收敛
        for i in range(self.__sift_img_border):
            img_current = np.float32(current_pyramid_DOG[int(layer)])
            img_prev = np.float32(current_pyramid_DOG[int(layer - 1)])
            img_next = np.float32(current_pyramid_DOG[int(layer + 1)])

            D = np.zeros((3, 1), dtype=np.float32)
            D[0, 0] = (img_current[r + 1, c] - img_current[r - 1, c]) /2.0  * img_scale    #dx
            D[1, 0] = (img_current[r, c + 1] - img_current[r, c - 1]) / 2.0 * img_scale    #dy
            D[2, 0] = (img_next[r, c] - img_prev[r, c]) / 2.0 * img_scale                  #ds

            dxx = (img_current[r + 1, c] + img_current[r - 1, c] - 2 * img_current[r, c]) * img_scale
            dyy = (img_current[r, c + 1] + img_current[r, c - 1] - 2 * img_current[r, c]) * img_scale
            dss = (img_next[r, c] + img_prev[r, c] - 2 * img_current[r, c]) * img_scale
            dxy = (img_current[r + 1, c + 1] + img_current[r - 1, c - 1] - img_current[r - 1, c + 1] - img_current[r + 1, c - 1]) /4.0 * img_scale
            dxs = (img_next[r + 1, c] + img_prev[r - 1, c] - img_next[r - 1, c]  - img_prev[r + 1, c]) /4.0 * img_scale
            dys = (img_next[r, c + 1] + img_prev[r, c - 1] - img_next[r, c - 1]  - img_prev[r, c + 1]) /4.0 * img_scale

            HD = np.array(([dxx, dxy, dxs],
                           [dxy, dyy, dys],
                           [dxs, dys, dss]), dtype=np.float32)

            X = HD.dot(D)
            xr, xc, xs = -X
            if np.abs(xr) < 0.5 or np.abs(xc) < 0.5 or np.abs(xs) < 0.5:
                break

            # (r, c, layer) + (xr, xc, xs)就是新的极值点
            r = int(r + np.round(xr))
            c = int(c + np.round(xc))
            layer = int(layer + np.round(xs))

            # 如果超出范围，直接停止插值好了
            if not 0 <= layer < octaves_layer_num \
                and not self.__sift_img_border <= r < img_current.shape[0] - self.__sift_img_border \
                and not self.__sift_img_border <= c < img_current.shape[1] - self.__sift_img_border:
                return False

        img_current = np.float32(current_pyramid_DOG[int(layer)])
        img_prev = np.float32(current_pyramid_DOG[int(layer - 1)])
        img_next = np.float32(current_pyramid_DOG[int(layer + 1)])

        D = np.zeros((1, 3), dtype=np.float32)
        D[0, 0] = (img_current[r + 1, c] - img_current[r - 1, c]) / 2.0 * img_scale  # dx
        D[0, 1] = (img_current[r, c + 1] - img_current[r, c - 1]) / 2.0 * img_scale  # dy
        D[0, 2] = (img_next[r, c] - img_prev[r, c]) / 2.0 * img_scale  # ds

        # 剔除低对比度的特征点
        # 这一步就是把(xr, xc, xs)代进去泰勒展开那个公式里面去(保留两项)
        D_max = img_current[r, c] * img_scale + 1 / 2 * D.dot(np.array([xr, xc, xs], dtype=np.float32))

        # 我也不知道为什么要乘以octaves_layer_num ?
        if np.abs(D_max) * octaves_layer_num < self.__extreme_point_value_threshold:
            return False

        # 除不稳定的边缘响应点
        v2 = 2 * img_current[r, c]
        dxx = (img_current[r + 1, c] + img_current[r - 1, c] - v2) * img_scale
        dyy = (img_current[r, c + 1] + img_current[r, c - 1] - v2) * img_scale
        dxy = (img_current[r + 1, c + 1] + img_current[r - 1, c - 1] - img_current[r - 1, c + 1] - img_current[
            r + 1, c - 1]) / 4.0 * img_scale

        tr = dxx + dyy #hessian矩阵的迹
        det = dxx * dyy - 2 * dxy

        curv_thr = self.__sift_contour_threshold

        # 主曲率大于(y + 1)^2 / y的时候偶，剔除掉
        if det <= 0 or tr**2 / det >= (curv_thr + 1)**2 / curv_thr:
            return False

        key_point = (r, c, layer, octave, 2**(layer / octaves_layer_num))
        key_point_array.append(key_point)
        return True

    @MethodInformProvider
    def __AccurateKeyPointLocalization(self, imgSrc, pyramid_DOG_list, octaves_layer_num, octaves_num)->list:
        border = self.__sift_img_border
        threshold = self.__extreme_point_value_threshold

        key_point_array = []
        for octave in range(octaves_num):
            for layer in range(1, octaves_layer_num - 1):
                h, w = pyramid_DOG_list[octave][layer].shape
                for r, c in zip(range(border, h - border), range(border, w - border)):
                    # 是否是极值点
                    if not self.__isMaximumPoint(pyramid_DOG_list ,octave, layer, r, c, threshold):
                        continue
                    vector = [r, c, layer]
                    if not self.__adjustLocalExtrema(pyramid_DOG_list, octave, vector, octaves_layer_num, key_point_array):
                        continue
        return key_point_array

    @MethodInformProvider
    def GetFeatures(self, init_sigma = 0.5, octaves_layer_num = 5, octaves_num = 3):
        imgSrc = self.__imgSrc
        if np.shape(imgSrc)[2] == 3:
            imgSrc = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)

        pyramid_DOG_list = self.__BuildDogPyramid(imgSrc, init_sigma, octaves_layer_num, octaves_num)
        key_point_array = self.__AccurateKeyPointLocalization(imgSrc, pyramid_DOG_list, octaves_layer_num, octaves_num)
        print(key_point_array)

        '''
        i = 0
        for octave in range(octaves_num):
            for layer in range(octaves_layer_num + 2):
                plt.figure(i)
                plt.imshow(pyramid_DOG_list[octave][layer], cmap="gray")
                i += 1
        plt.show()
        '''

        imgDst = self.__imgSrc
        for point in key_point_array:
            imgDst = cv.circle(imgDst, point[:2], 5, (0, 0, 255))

        imgDst = cv.cvtColor(imgDst, cv.COLOR_BGR2RGB)
        plt.imshow(imgDst)
        plt.show()

        '''
       
        '''



