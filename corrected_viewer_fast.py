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
mtx = np.array([[1, 0, 320], [0, 1, 240], [0, 0, 1]])
dist = np.array([1.05791597e-06,  5.26154073e-14, -4.30326545e-06, -4.60648477e-06,
                 0, 3.41991153e-06, 3.27612688e-13, 0])
newcameramtx = np.loadtxt(
    "newcameramtx.csv", delimiter=",", skiprows=0, usecols=(0, 1, 2))


def nothing(x):
    pass


def update_image():
    global img, corrected_img, roi, mtx, dist, newcameramtx
    corrected_img = cv2.undistort(img, mtx, dist)
    cv2.imshow(corrected_image_window, corrected_img)


def change_image(val):
    global img_no, img
    img_no = val
    img = cv2.imread(get_img_name(img_no))
    cv2.imshow(raw_image_window, img)
    update_image()


if __name__ == "__main__":

    # 画像の表示
    change_image(img_no)

    cv2.createTrackbar("Image No", raw_image_window, img_no, 202, change_image)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
