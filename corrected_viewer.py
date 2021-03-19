# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_img_name(no):
    return "frame"+str(no).zfill(4)+".jpg"


# paramter
raw_image_window = "raw image"
corrected_image_window = "corrected image"
img_no = 0
img = cv2.imread(get_img_name(0))
corrected_img = img.copy()
mtx = np.loadtxt("mtx.csv", delimiter=",", skiprows=0, usecols=(0, 1, 2))
dist = np.loadtxt("dist.csv", delimiter=",",
                  skiprows=0, usecols=(0, 1, 2, 3, 4, 5))


def nothing(x):
    pass


def update_image():
    global img, corrected_img, mtx, dist
    corrected_img = np.zeros((len(img), len(img[0]), 3), np.uint8)
    [k1, k2, k3, k4, p1, p2] = dist
    for i in range(len(img)):
        for j in range(len(img[0])):
            x1 = j - 320
            y1 = i - 240
            r_2 = x1 ** 2 + y1 ** 2
            r_4 = r_2 ** 2
            x2 = x1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * p1 * \
                x1 * y1 + p2 * (r_2 + 2 * x1 ** 2)
            y2 = y1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * p2 * \
                x1 * y1 + p1 * (r_2 + 2 * y1 ** 2)
            ii = int(y2 + 240)
            jj = int(x2 + 320)
            if ii < 0 or ii >= 480:
                continue
            if jj < 0 or jj >= 640:
                continue
            corrected_img[i][j] = img[ii][jj]
    cv2.imshow(corrected_image_window, corrected_img)


def change_image(val):
    global img_no, img
    img_no = val
    img = cv2.imread(get_img_name(img_no))
    cv2.imshow(raw_image_window, img)
    update_image()


if __name__ == "__main__":
    print(dist)
    # 画像の表示
    change_image(img_no)

    cv2.createTrackbar("Image No", raw_image_window, img_no, 202, change_image)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
