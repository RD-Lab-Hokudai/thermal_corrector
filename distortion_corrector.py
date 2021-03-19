# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


def get_img_name(no):
    return "frame"+str(no).zfill(4)+".jpg"


# paramter
window_name = "input window"
img_no = 0
is_reverse = True
lower = 160
upper = 255
i = 20
k = 5

img = cv2.imread(get_img_name(img_no))
h, w = img.shape[:2]
left = 0
right = w-1
up = 0
down = h-1
output = img.copy()
row = 1
collumn = 1
obj_points = []
img_points = []
tmp_obj_points = []
tmp_img_points = []

if os.path.exists("obj_points_tmp.txt"):
    with open("obj_points_tmp.txt") as f:
        points = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "[":
                points = []
            elif line == "]":
                obj_points.append(np.array(points, dtype='float32'))
            else:
                points.append([float(val)
                               for val in line[1:len(line) - 2].split(",")])
print(obj_points)

if os.path.exists("img_points_tmp.txt"):
    with open("img_points_tmp.txt") as f:
        points = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "[":
                points = []
            elif line == "]":
                img_points.append(np.array(points, dtype='float32'))
            else:
                points.append([[float(val)
                                for val in line[2:len(line) - 3].split(",")]])
print(img_points)


def nothing(x):
    pass


def update_output():
    global img, output, is_reverse, lower, upper, i, k
    output = img.copy()

    if is_reverse:
        output = ~output

    kernel = -np.ones((i, i), np.float32) / (i * i) * k
    kernel[int(i/2), int(i/2)] = kernel[int(i/2), int(i/2)]+k+1
    output = cv2.filter2D(output, -1, kernel)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output = cv2.inRange(output, np.array(
        [0, 0, lower]), np.array([0, 0, upper]))

    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    # output=cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
    cv2.rectangle(output, (left, up), (right, down), (250, 0, 0))

    cv2.imshow(window_name, output)


def change_image(val):
    global img_no, img
    img_no = val
    img = cv2.imread(get_img_name(img_no))
    update_output()


def reverse(val):
    global is_reverse
    is_reverse = val
    update_output()


def change_i(val):
    global i
    i = val
    update_output()


def change_k(val):
    global k
    k = val
    update_output()


def change_lower(val):
    global lower
    lower = val
    update_output()


def change_upper(val):
    global upper
    upper = val
    update_output()


def change_left(val):
    global left
    left = val
    update_output()


def change_right(val):
    global right
    right = val
    update_output()


def change_up(val):
    global up
    up = val
    update_output()


def change_down(val):
    global down
    down = val
    update_output()


def change_row(val):
    global row
    row = val


def change_collumn(val):
    global collumn
    collumn = val


def find_contours(val):
    if val == 0:
        return

    global output, left, right, up, down, row, collumn, tmp_obj_points, tmp_img_points
    rectangle = output[up+1:down-1, left+1:right-1]
    gray = cv2.cvtColor(rectangle, cv2.COLOR_BGR2GRAY)
    cntImage, contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_centers = []
    for cnts in contours:
        area = cv2.contourArea(cnts)
        moments = cv2.moments(cnts)
        if moments['m00'] > 1e-9:
            cx = float(moments['m10']/moments['m00'])
            cy = float(moments['m01']/moments['m00'])
            filtered_centers.append([[cx, cy]])

    filtered_centers = sorted(filtered_centers, key=lambda x: x[0][0])
    temp_centers = []
    sorted_centers = []
    for i in range(len(filtered_centers)):
        temp_centers.append(filtered_centers[i])
        if i % row == row-1:
            sorted_centers += sorted(temp_centers, key=lambda x: x[0][1])
            temp_centers = []

    sorted_centers = np.array(sorted_centers, dtype='float32')
    img = cv2.drawChessboardCorners(
        rectangle, (row, collumn), sorted_centers, True)
    #img=cv2.drawContours(rectangle, contours, 3, (0,255,0), 3)
    output[up+1:down-1, left+1:right-1] = img
    cv2.imshow(window_name, output)

    tmp_obj_points = np.zeros((row*collumn, 3), np.float32)
    tmp_obj_points[:, :2] = np.mgrid[0:row, 0:collumn].T.reshape(-1, 2)
    sorted_centers += np.array([[left+1, up+1]])
    tmp_img_points = sorted_centers


def append_points(val):
    if val == 0:
        return

    global obj_points, img_points, tmp_obj_points, tmp_img_points
    obj_points.append(tmp_obj_points)
    img_points.append(tmp_img_points)
    print(obj_points)
    print(img_points)


def save_obj_points():
    s = ""
    for i in range(0, len(obj_points)):
        s += "[\n"
        for point in obj_points[i]:
            s += "[" + str(point[0]) + "," + str(point[1]) + \
                "," + str(point[2]) + "],\n"
        s += "]"
        if i < len(obj_points) - 1:
            s += "\n"

    with open("obj_points_tmp.txt", mode="w") as f:
        f.write(s)


def save_img_points():
    s = ""
    for i in range(0, len(img_points)):
        s += "[\n"
        for point in img_points[i]:
            s += "[[" + str(point[0][0]) + "," + str(point[0][1]) + "]],\n"
        s += "]"
        if i < len(img_points) - 1:
            s += "\n"

    with open("img_points_tmp.txt", mode="w") as f:
        f.write(s)


def correct_distortion(val):
    if val == 0:
        return

    global output, obj_points, img_points
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
    h, w = output.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    output = cv2.undistort(output, mtx, dist, None, newcameramtx)

    if roi[2] == 0 or roi[3] == 0:
        print("cant correct!")
        obj_points.pop(len(obj_points)-1)
        img_points.pop(len(img_points)-1)
        print(obj_points)
        return

    x, y, w, h = roi
    output = output[y:y+h, x:x+w]
    cv2.imshow(window_name, output)
    print((x, y, w, h))
    print(mtx)
    print(dist)
    print(newcameramtx)
    save_obj_points()
    save_img_points()
    np.savetxt("roi.csv", (x, y, w, h), delimiter=',')
    np.savetxt('mtx.csv', mtx, delimiter=',')
    np.savetxt('dist.csv', dist, delimiter=',')
    np.savetxt('newcameramtx.csv', newcameramtx, delimiter=',')


if __name__ == "__main__":

    # 画像の表示
    update_output()

    cv2.createTrackbar("Reverse", window_name, is_reverse, 1, reverse)
    cv2.createTrackbar("Image No", window_name, img_no, 202, change_image)

    cv2.createTrackbar("I", window_name, i, 255, change_i)
    cv2.createTrackbar("K", window_name, k, 255, change_k)

    cv2.createTrackbar("Lower", window_name, lower, 255, change_lower)
    cv2.createTrackbar("Upper", window_name, upper, 255, change_upper)

    cv2.createTrackbar("Left", window_name, left, w, change_left)
    cv2.createTrackbar("Right", window_name, right, w, change_right)
    cv2.createTrackbar("Up", window_name, up, h, change_up)
    cv2.createTrackbar("Down", window_name, down, h, change_down)

    cv2.createTrackbar("Row", window_name, row, 20, change_row)
    cv2.createTrackbar("Collumn", window_name, collumn, 20, change_collumn)

    cv2.createTrackbar("Find Countours", window_name, 0, 1, find_contours)

    cv2.createTrackbar("Append Points", window_name, 0, 1, append_points)

    cv2.createTrackbar("Correct Distortion", window_name,
                       0, 1, correct_distortion)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
