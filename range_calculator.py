
import numpy as np
import cv2
import matplotlib.pyplot as plt

dist = np.loadtxt(
    "dist.csv", delimiter=",", skiprows=0, usecols=(0, 1, 2, 3, 4, 5))
mtx = np.array([[1, 0, 320], [0, -1, 320], [0, 0, 1]])


def fit_current(param, x1, y1):
    [k1, k2, k4, k5, p1, p2] = param
    r_2 = x1 ** 2 + y1 ** 2
    r_4 = r_2 ** 2
    x2 = x1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k4 * r_2 + k5 *
                                           r_4) + 2 * p1 * x1 * y1 + p2 * (r_2 + 2 * x1 ** 2)
    y2 = y1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k4 * r_2 + k5 *
                                           r_4) + 2 * p2 * x1 * y1 + p1 * (r_2 + 2 * y1 ** 2)
    return x2, y2


if __name__ == '__main__':
    image = np.ones((480, 640))
    height = 1000
    width = 1500
    corrected = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x1 = j-width/2
            y1 = height / 2 - i
            x2, y2 = fit_current(dist, x1, y1)
            jj = int(x2 + 320)
            ii = int(240 - y2)
            if 0 <= ii and ii < 480 and 0 <= jj and jj < 640:
                corrected[i][j] = image[ii][jj]
    for i in range(height):
        if corrected[i][int(width / 2)] == 1:
            print(i)
            break
    for i in range(height):
        if corrected[height-i-1][int(width / 2)] == 1:
            print(height-i-1)
            break
    for j in range(width):
        if corrected[int(height/2)][j] == 1:
            print(j)
            break
    for j in range(width):
        if corrected[int(height/2)][width-j-1] == 1:
            print(width-j-1)
            break
    cv2.imshow('range', corrected[215:781, 307:1206])
    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
