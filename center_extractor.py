import sys
import numpy as np
import cv2

image = np.zeros((1, 1, 3), np.uint8)
output = np.zeros((1, 1, 3), np.uint8)
lower = 0
upper = 254


def update():
    global image, output, lower, upper, win_name
    output = np.zeros(image.shape, dtype='uint8')
    color = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(
        color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    for contour in contours:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        if (0 <= x and x < 640 and 0 <= y and y < 480):
            output[int(y)][int(x)] = 255
    cv2.imshow('points', output)
    cv2.imshow("range", color)


def change_lower(val):
    global lower
    lower = val
    update()


def change_upper(val):
    global upper
    upper = val
    update()


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print('No image file is specified!')
        exit()

    image = cv2.imread(args[1])
    if image is None:
        print('Invalid image format!')
        exit()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = image.copy()

    update()
    cv2.createTrackbar('Lower', 'range', lower, 255, change_lower)
    cv2.createTrackbar('Upper', 'range', upper, 255, change_upper)
    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
