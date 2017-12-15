import cv2 as cv


def parse_bit(src,level):
    dst = src.copy()
    for line in dst:
        for col in line:
            byte_color = col[0]
            mask = 1 << level
            byte_color = byte_color & mask
            new_color = byte_color == mask and 0xff or 0
            col[0] = new_color
            col[1] = new_color
            col[2] = new_color
    str_win_name = "Image%d" %level
    cv.namedWindow(str_win_name)
    cv.imshow(str_win_name, dst)


img = cv.imread("E:/Users/Administrator/pictures/Test/dollar.jpg")

for i in range(8):
    parse_bit(img, i)

cv.namedWindow("Image")
cv.imshow("Image",img)
cv.waitKey(0)

cv.destroyAllWindows()