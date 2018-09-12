from common import *
from matplotlib import pyplot as plt
from numba import jit

def _gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv.getGaussianKernel(kernel_size, sigma)
    ky = cv.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

class SiftFeature2D:
    def __init__(self, imgSrc:np.ndarray):
        self.__extreme_point_value_threshold = 0.04
        self.__imgSrc:np.ndarray = imgSrc
        self.__sift_img_border = 5
        self.__sift_contour_threshold = 10.0
        self.__ori_peak_ratio = 0.8
        self.__ori_hist_bins = 36
        self.__ori_sig_fact = 1.5

        self.__region_size = 4                  # 划分区域大小
        self.__region_ori_calc = 8              # 每个区域要计算的方向个数 360 / k = angle

    def __BuildDogPyramid(self, imgSrc, init_sigma, octaves_layer_num ,octaves_num):
        k = 2**(1 / octaves_num)
        sigma = np.zeros([octaves_layer_num + 3], dtype=np.float64)
        sigma[0] = init_sigma

        '''
        for i in range(1, octaves_layer_num + 3):
            prev_sigma = init_sigma*(k**(i - 1))
            cur_sigma = prev_sigma*k
            sigma[i] = np.sqrt(cur_sigma**2 - prev_sigma**2)
            # sigma[i] = cur_sigma
        '''
        sigma[1] = init_sigma * np.sqrt(k * k - 1)
        for i in range(2, octaves_layer_num + 3):
            sigma[i] = sigma[i - 1] * k
            # sigma[i] = cur_sigma

        pyramid_list = []
        cur_height, cur_width = imgSrc.shape[:2]
        # 先构造高斯金字塔
        for octave in range(octaves_num):
            pyramid_list.append(np.zeros([octaves_layer_num + 3, cur_height, cur_width], dtype=np.float32))
            current_layer_pack = pyramid_list[octave]
            for layer in range(octaves_layer_num + 3):
                if octave == 0 and layer == 0:
                    # 第0层第0张图片，当然是原图了
                    current_layer_pack[layer] = imgSrc
                elif layer == 0:
                    # 每一组第0副图像时上一组倒数第三幅图像隔点采样得到（尺度刚好是2 sigma）
                    current_layer_pack[layer] = cv.resize(pyramid_list[octave - 1][octaves_layer_num],
                                                          dsize=(cur_width, cur_height), interpolation=cv.INTER_CUBIC)
                else:
                    prev_img = current_layer_pack[layer - 1]
                    current_layer_pack[layer] = self.__Gaussian(prev_img, sigma[layer])

            cur_height = cur_height // 2
            cur_width = cur_width // 2

        # 构造DOG金字塔，由高斯金字塔两两相减
        cur_height, cur_width = imgSrc.shape[:2]
        pyramid_DOG_list = []
        for octave in range(octaves_num):
            pyramid_DOG_list.append(np.zeros([octaves_layer_num + 3, cur_height, cur_width], dtype=np.float32))

            current_layer_pack = pyramid_list[octave]
            current_DOG_layer_pack = pyramid_DOG_list[octave]
            for layer in range(octaves_layer_num + 2):
                first_mat = current_layer_pack[layer]
                second_mat = current_layer_pack[layer + 1]
                result = np.subtract(second_mat, first_mat)
                current_DOG_layer_pack[layer] = result

            cur_height = cur_height // 2
            cur_width = cur_width // 2

        return pyramid_DOG_list

    @staticmethod
    def __Gaussian(img, sigma):
        k = 3
        k_size = round(2 * k * sigma + 1)
        if k_size % 2 == 0:
            k_size = k_size + 1
        kernel = _gaussian_kernel_2d_opencv(int(k_size), sigma)
        out_img = cv.filter2D(img, -1, cv.flip(kernel, -1), borderType=cv.BORDER_CONSTANT)
        return out_img

    @staticmethod
    def __IsMaximumPoint(pyramid_DOG_list, octave, layer, r, c, threshold):
        img_current = pyramid_DOG_list[octave][layer]
        img_prev = pyramid_DOG_list[octave][layer - 1]
        img_next = pyramid_DOG_list[octave][layer + 1]

        value = img_current[r, c]
        img_block_current = img_current[r - 1:r + 2, c - 1:c + 2]
        img_block_prev = img_prev[r - 1:r + 2, c - 1:c + 2]
        img_block_next = img_next[r - 1:r + 2, c - 1:c + 2]

        if np.abs(value) < threshold:
            return False

        if value > 0 and value >= np.max(img_block_current) and value >= np.max(img_block_prev) and value >= np.max(img_block_next):
            return True

        return value <= np.min(img_block_current) and value <= np.min(img_block_prev) and value <= np.min(img_block_next)

    def __AdjustLocalExtrema(self, pyramid_DOG_list, octave, vector, threshold, octaves_layer_num, octaves_num, key_point_array: list):
        """
        插值和删除边缘效应
        :return:
        """
        current_pyramid_DOG = pyramid_DOG_list[octave]
        r, c, layer = vector
        xr, xc, xs = (0.0, 0.0, 0.0)
        # 插值最大次数，防止不收敛
        for i in range(self.__sift_img_border + 1):
            if i == self.__sift_img_border:
                return False

            img_current = current_pyramid_DOG[int(layer)]
            img_prev = current_pyramid_DOG[int(layer - 1)]
            img_next = current_pyramid_DOG[int(layer + 1)]

            D = np.zeros((3, 1), dtype=np.float32)
            D[0, 0] = (img_current[r + 1, c] - img_current[r - 1, c]) / 2.0     #dx
            D[1, 0] = (img_current[r, c + 1] - img_current[r, c - 1]) / 2.0     #dy
            D[2, 0] = (img_next[r, c] - img_prev[r, c]) / 2.0                   #ds

            v2 = 2 * img_current[r, c]
            dxx = (img_current[r + 1, c] + img_current[r - 1, c] - v2)
            dyy = (img_current[r, c + 1] + img_current[r, c - 1] - v2)
            dss = (img_next[r, c] + img_prev[r, c] - v2)
            dxy = (img_current[r + 1, c + 1] + img_current[r - 1, c - 1] - img_current[r - 1, c + 1] - img_current[r + 1, c - 1]) /4.0
            dxs = (img_next[r + 1, c] + img_prev[r - 1, c] - img_next[r - 1, c]  - img_prev[r + 1, c]) /4.0
            dys = (img_next[r, c + 1] + img_prev[r, c - 1] - img_next[r, c - 1]  - img_prev[r, c + 1]) /4.0

            HD = np.array(([dxx, dxy, dxs],
                           [dxy, dyy, dys],
                           [dxs, dys, dss]))

            X = HD.dot(D)
            xr, xc, xs = -X
            if np.abs(xr) < 0.5 and np.abs(xc) < 0.5 and np.abs(xs) < 0.5:
                break

            # (r, c, layer) + (xr, xc, xs)就是新的极值点
            r = int(r + np.round(xr))
            c = int(c + np.round(xc))
            layer = int(layer + np.round(xs))

            # 如果超出范围，直接停止插值好了
            if not 1 <= layer < octaves_layer_num - 1 \
                or not self.__sift_img_border <= r < img_current.shape[0] - self.__sift_img_border \
                or not self.__sift_img_border <= c < img_current.shape[1] - self.__sift_img_border:
                return False

        img_current = current_pyramid_DOG[int(layer)]
        img_prev = current_pyramid_DOG[int(layer - 1)]
        img_next = current_pyramid_DOG[int(layer + 1)]

        D = np.zeros((1, 3), dtype=np.float32)
        D[0, 0] = (img_current[r + 1, c] - img_current[r - 1, c]) / 2.0  # dx
        D[0, 1] = (img_current[r, c + 1] - img_current[r, c - 1]) / 2.0  # dy
        D[0, 2] = (img_next[r, c] - img_prev[r, c]) / 2.0  # ds

        # 剔除低对比度的特征点
        # 这一步就是把(xr, xc, xs)代进去泰勒展开那个公式里面去(保留两项)
        D_max = img_current[r, c] + 1 / 2 * D.dot(np.array([xr, xc, xs], dtype=np.float32))

        # 我也不知道为什么要乘以octaves_num ?
        if np.abs(D_max) * octaves_num < threshold:
            return False

        # 除不稳定的边缘响应点
        v2 = 2 * img_current[r, c]
        dxx = (img_current[r + 1, c] + img_current[r - 1, c] - v2)
        dyy = (img_current[r, c + 1] + img_current[r, c - 1] - v2)
        dxy = (img_current[r + 1, c + 1] + img_current[r - 1, c - 1] - img_current[r - 1, c + 1] - img_current[r + 1, c - 1]) / 4.0

        tr = dxx + dyy #hessian矩阵的迹
        det = dxx*dyy - dxy**2

        curv_thr = self.__sift_contour_threshold

        # 主曲率大于(y + 1)^2 / y的时候要剔除掉这个点
        if det <= 0 or tr**2 / det >= (curv_thr + 1)**2 / curv_thr:
            return False

        key_point = (r, c, layer, octave, 2**(layer / octaves_layer_num))
        key_point_array.append(key_point)
        return True

    @MethodInformProvider
    def __AccurateKeyPointLocalization(self, pyramid_DOG_list, octaves_layer_num, octaves_num)->list:
        border = self.__sift_img_border
        threshold = self.__extreme_point_value_threshold / octaves_num * 1/2

        key_point_array = []
        for octave in range(octaves_num):
            h, w = pyramid_DOG_list[octave][0].shape
            for layer in range(1, octaves_layer_num - 1):
                for r in range(border, h - border):
                    for c in range(border, w - border):
                        # 是否是极值点
                        img = pyramid_DOG_list[octave][layer]
                        if not self.__IsMaximumPoint(pyramid_DOG_list , octave, layer, r, c, threshold):
                            continue
                        vector = [r, c, layer]
                        if not self.__AdjustLocalExtrema(pyramid_DOG_list, octave, vector,
                                                         self.__extreme_point_value_threshold,
                                                         octaves_layer_num, octaves_num, key_point_array):
                            continue

        return key_point_array

    @staticmethod
    def __GetGradient(imgSrc, r, c):
        height, width = imgSrc.shape[:2]

        assert 1 <= c < width and 1 <= r <height
        dy = imgSrc[r + 1, c] - imgSrc[r - 1, c]
        dx = imgSrc[r, c + 1] - imgSrc[r, c - 1]

        m = np.sqrt(dx**2 + dy**2)
        theta =  np.arctan2(dx, dy)

        return m, theta

    # 论文提到的对直方图进行插值的方法
    @staticmethod
    def __HistogramInterp(position, left, cur, right):
        position += 0.5 * (left - right) / (left - 2 * cur + right)
        return position

    def __CaculateHistogram(self, imgSrc, r, c, radius, sigma):
        histogram = np.zeros(self.__ori_hist_bins, dtype=np.float64)
        h, w = imgSrc.shape[:2]
        exp_sigma = 2*sigma*sigma

        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if r + i < 0 or r + i > h or c + j < 0 or c + j > w:
                    continue
                m, theta = self.__GetGradient(imgSrc, r + i, c + j)
                w = np.exp((i*i +j*j)/exp_sigma)

                # 让范围落在self.__ori_hist_bins之内
                angle = int(round(self.__ori_hist_bins*(theta + np.pi) / (2*np.pi)) % self.__ori_hist_bins)
                angle %= self.__ori_hist_bins
                histogram[angle] += w*m

        temp_histogram = histogram.copy()

        # 对直方图进行平滑处理
        total_size = len(temp_histogram)
        for i in range(self.__ori_hist_bins):
            histogram[i] = 1 / 16 *(temp_histogram[(i - 2) % total_size] + temp_histogram[(i + 2)% total_size]) \
                         + 4 / 16 *(temp_histogram[(i - 1)% total_size] + temp_histogram[(i + 1)% total_size]) \
                         + 6 * temp_histogram[i % total_size] / 16

        return histogram

    @MethodInformProvider
    def __CalculateOrientationHist(self, pyramid_DOG_list, key_point_array):
        ori_sig_fact = self.__ori_sig_fact

        feature_array = []
        for key_point in key_point_array:
            r, c, layer, octave, scale = key_point

            # 每一个关键点围绕他的1.5*3*尺度的半径进行寻找
            sigma = ori_sig_fact * scale
            radius = int(3 * sigma)
            histogram = self.__CaculateHistogram(pyramid_DOG_list[octave][layer], r, c, radius, sigma)

            max_val = np.max(histogram)

            # 添加主方向和辅助方向（大于主峰80%）
            for i in range(len(histogram)):
                cur = histogram[i]
                if i == 0:
                    left = histogram[-1]
                    right = histogram[1]
                elif i == len(histogram) - 1:
                    left = histogram[i - 1]
                    right = histogram[0]

                if np.max((left, cur, right)) != cur:
                    continue
                if cur < max_val * self.__ori_peak_ratio:
                    continue
                pos = self.__HistogramInterp(i, left, cur, right)

                angle = pos / self.__ori_hist_bins * 2 *np.pi - np.pi
                feature = (key_point, angle)
                feature_array.append(feature)

        return feature_array

    @staticmethod
    def __InterpolateHist(hist, r_bin, c_bin, theta_norm, m, descriptor_block_width, descriptor_ori_sum):
        r0 = int(np.floor(r_bin))
        c0 = int(np.floor(c_bin))
        o0 = int(np.floor(theta_norm))
        d_r = r_bin - r0
        d_c = c_bin - c0
        d_o = theta_norm - o0

        for i in range(1):
            if r0 + i >= descriptor_block_width or r0 + i < 0:
                continue
            for j in range(1):
                if c0 + j >= descriptor_block_width or c0 + j < 0:
                    continue
                for k in range(1):
                    theta_norm_index = (theta_norm + k) % descriptor_ori_sum

                    # 双线性插值
                    # 下面这个写的好像很复杂，其实你把0和1代进去就看出来是什么了...，就是(1- x)和x的组合
                    value = m * ( 0.5 + (d_r - 0.5)*(2*i-1) ) \
                                * ( 0.5 + (d_c - 0.5)*(2*j-1) ) \
                                * ( 0.5 + (d_o - 0.5)*(2*k-1) )

                    hist_index = int((r0 + i)*descriptor_block_width*descriptor_ori_sum
                                 + (c0 + j)*descriptor_block_width + theta_norm_index + k)

                    hist[hist_index] = hist[hist_index] + value

    @staticmethod
    def __RemoveLightEffect(hist, descriptor_mag_thr):
        hist = hist / np.linalg.norm(hist)
        hist = np.where(hist >= descriptor_mag_thr, hist, 0)
        hist = hist / np.linalg.norm(hist)
        return hist

    @MethodInformProvider
    def __DescriptorGeneration(self, pyramid_DOG_list, feature_array, init_sigma):
        m = 3
        descriptor_block_num = 4
        descriptor_block_width = 4
        descriptor_ori_sum = 8
        # 8个方向，4*4个区域，一共128维
        descriptor_size = descriptor_block_width*descriptor_block_width*descriptor_ori_sum
        descriptor_mag_thr = 0.2

        descriptor_list = []

        for item in feature_array:
            key_point, angle = item
            r, c, layer, octave, oct_sigma = key_point

            img = pyramid_DOG_list[octave][layer][r,c]
            # 4*4个区域，每个区域宽度为4，考虑到双线性插值，所以d+ 1，另外再考虑到旋转，要包括45度的情况
            radius = int(np.round(descriptor_block_num * (descriptor_block_width + 1) *np.sqrt(2) / 2))
            hist_width = init_sigma*oct_sigma*m

            hist = np.zeros(descriptor_size, dtype=np.float32)
            for i in range(int(-radius), int(radius)):
                for j in range(int(-radius), int(radius)):
                    '''
                    # Calculate sample's histogram array coords rotated relative to ori.
                    # Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
                    # r_rot = 1.5) have full weight placed in row 1 after interpolation.
                    '''
                    i_rot = i*np.cos(angle) - j*np.sin(angle)
                    j_rot = j*np.cos(angle) + i*np.sin(angle)
                    r_bin = i_rot / hist_width + descriptor_block_width / 2 - 0.5
                    c_bin = j_rot / hist_width + descriptor_block_width / 2 - 0.5
                    if -1 < r_bin < descriptor_block_width and -1 < c_bin < descriptor_block_width:
                        h, w = img.shape[:2]
                        if r + i < 0 or r + i > h or c + j < 0 or c + j > w:
                            continue
                        mag, theta = self.__GetGradient(img, r, c)
                        theta = theta - angle   # 把角度转回来
                        if theta < 0:
                            theta = theta + 2 * np.pi
                        else:
                            theta = theta - 2 * np.pi

                        # 把角度约束在8个方向内
                        theta_norm = theta * descriptor_ori_sum/ (2*np.pi)
                        w = -(i_rot**2 + j_rot**2) / (2 *hist_width**2)
                        self.__InterpolateHist(hist, r_bin, c_bin, theta_norm,
                                               mag*w, descriptor_block_width, descriptor_ori_sum)

            hist = self.__RemoveLightEffect(hist, descriptor_mag_thr)

            descriptor_list.append((item, hist))

        sorted(descriptor_list, key=lambda k: k[0][0][4])
        return descriptor_list


    @MethodInformProvider
    def GetFeatures(self, init_sigma = 1.6, octaves_layer_num = 5, octaves_num = 3):
        imgSrc = self.__imgSrc
        if len(np.shape(imgSrc)) == 3:
            imgSrc = cv.cvtColor(imgSrc, cv.COLOR_BGR2GRAY)
        imgSrc = imgSrc.astype(np.float32) / 255.0
        imgSrc = cv.resize(imgSrc, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
        imgSrc = self.__Gaussian(imgSrc, np.sqrt(init_sigma ** 2 - 0.5 ** 2 * 4))

        # 先对图片从uint8转成flaot，归一化

        pyramid_DOG_list = self.__BuildDogPyramid(imgSrc, init_sigma, octaves_layer_num, octaves_num)
        key_point_array = self.__AccurateKeyPointLocalization(pyramid_DOG_list, octaves_layer_num, octaves_num)
        feature_array = self.__CalculateOrientationHist(pyramid_DOG_list, key_point_array)

        self.__DescriptorGeneration(pyramid_DOG_list, feature_array, init_sigma)

        print(len(key_point_array))
        imgDst = self.__imgSrc
        for feature in feature_array:
            point = feature[0]
            h, w, _, octave = point[:4]

            real_w = int(round(w*(2**(octave - 1))))
            real_h = int(round(h*(2**(octave - 1))))
            cv.circle(imgDst,(real_w, real_h), 5, (255, 255,0))

            angle = feature[1]
            real_h_pt = int(real_h + 10 * np.sin(angle))
            real_w_pt = int(real_w + 10 * np.cos(angle))

            cv.arrowedLine(imgDst, (real_w, real_h), (real_w_pt, real_h_pt), (255, 255, 0))


        if len(np.shape(imgSrc)) == 3:
            imgDst = cv.cvtColor(imgDst, cv.COLOR_BGR2RGB)
            plt.imshow(imgDst)
            plt.show()
        elif len(np.shape(imgSrc)) == 2:
            plt.imshow(imgDst, cmap="gray")
            plt.show()
        else:
            raise NotImplementedError("wtf")
