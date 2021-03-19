
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import optimize


def annotation_array(column_len, column_count):
    columns = []
    for i in range(column_len):
        x = i * (2.5 * 3 ** 0.5)
        y = 0
        if i % 2 == 1:
            y += 2.5
        column = []
        for j in range(column_count):
            column.append((x, y))
            y += 5
        columns.append(column)
    return columns


def extract_points(points, annotation_begin, x_rev=False, y_rev=False, rotated=False):
    columns = []
    base_x = 310
    color_no = 0
    cm = plt.get_cmap('jet')
    annotations = annotation_array(40, 50)
    column_cnt_base_odd = 0
    column_cnt_base_even = 0
    for i in range(len(points)):
        if x_rev:
            points[i] = (639 - points[i][0], points[i][1])
        if y_rev:
            points[i] = (points[i][0], 479 - points[i][1])
        if rotated:
            points[i] = (320 + 240 - points[i][1], 240 + 320 - points[i][0])
    while (True):
        if len(points) == 0:
            break

        min_i = (0, 10000000000)
        for i in range(len(points)):

            dist = (points[i][0] - base_x) ** 2 + (points[i][1] - 240) ** 2
            if dist < min_i[1]:
                min_i = (i, dist)

        column = []
        max_x = points[min_i[0]][0]
        base_y = points[min_i[0]][1] + 1
        column_no = len(columns)
        upper_cnt = 0
        # bottom up annotation
        if column_no > 15:
            while upper_cnt < len(columns[column_no - 2]) and columns[column_no - 2][upper_cnt][3] > base_y:
                if column_no % 2 == 0:
                    column_cnt_base_even += 1
                else:
                    column_cnt_base_odd += 1
                upper_cnt += 1
                # print(len(columns[column_no-2]),
                #      column_cnt_base_even, column_cnt_base_odd)
        while (True):
            min_p = (-1, -1000000000)
            for i in range(len(points)):
                if points[i][0] <= max_x + 1 and points[i][1] < base_y:
                    if points[i][1] > min_p[1]:
                        min_p = (i, points[i][1])
            if min_p[0] == -1:
                break
            column_cnt = len(column)
            if column_no % 2 == 0:
                column_cnt += column_cnt_base_even
            else:
                column_cnt += column_cnt_base_odd
            # print((column_no, column_cnt))
            annotation = annotations[column_no+annotation_begin][column_cnt]
            column.append([annotation[0], annotation[1],
                           points[min_p[0]][0], points[min_p[0]][1]])
            max_x = points[min_p[0]][0]
            base_y = points[min_p[0]][1]

        color = cm(color_no+(color_no % 2)*64)
        for point in column:
            points.remove((point[2], point[3]))
        base_x = column[0][2]
        columns.append(column)
        color_no += 5
    for column in columns:
        for point in column:
            print(point)
            if rotated:
                [x, y, u, v] = point
                point[0] = y
                point[1] = x
                point[2] = 320 + 240 - v
                point[3] = 320 + 240 - u
            if x_rev:
                point[0] *= -1
                point[2] = 639 - point[2]
            if y_rev:
                point[1] *= -1
                point[3] = 479 - point[3]
            print(point)
    return columns


def fit(param, x):
    [d, k1, k2, _, k4, k5, _, p1, p2, alpha] = param
    x1 = d * x[0]
    y1 = d * x[1]*alpha
    x2 = x[2]
    y2 = x[3]
    r_2 = np.square(x1) + np.square(y1)
    r_4 = np.square(r_2)
    r_6 = ｒ_2 * r_4
    x_delta = np.square(x1 * (1 + k1 * r_2 + k2 * r_4 + 0 * r_6) / (1 + k4 * r_2 + k5 * r_4 + 0 * r_6) +
                        2 * p1 * x1 * y1 + p2 * (r_2 + 2 * np.square(x1)) - x2)
    y_delta = np.square(y1 * (1 + k1 * r_2 + k2 * r_4 + 0 * r_6) / (1 + k4 * r_2 + k5 * r_4 + 0 * r_6) +
                        2 * p2 * x1 * y1 + p1 * (r_2 + 2 * np.square(y1)) - y2)
    print(np.sum(np.sqrt(x_delta+y_delta))/len(x[0]))
    return x_delta + y_delta


def fit_current(param, x1, y1):
    [k1, k2, k4, k5, p1, p2] = param
    r_2 = x1 ** 2 + y1 ** 2
    r_4 = r_2 ** 2
    x2 = x1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k4 * r_2 + k5 * r_4) + 2 * \
        p1 * x1 * y1 + p2 * (r_2 + 2 * x1 ** 2)
    y2 = y1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k4 * r_2 + k5 * r_4) + 2 * \
        p2 * x1 * y1 + p1 * (r_2 + 2 * y1 ** 2)
    return x2, y2


def fit_old(param, x1, y1):
    [k1, k2, k3, p1, p2] = param
    r_2 = x1 ** 2 + y1 ** 2
    r_4 = r_2 ** 2
    r_6 = r_2*r_4
    x2 = x1 * (1 + k1 * r_2 + k2 * r_4 + k3 * r_6) + 2 * \
        p1 * x1 * y1 + p2 * (r_2 + 2 * x1 ** 2)
    y2 = y1 * (1 + k1 * r_2 + k2 * r_4 + k3 * r_6) + 2 * \
        p2 * x1 * y1 + p1 * (r_2 + 2 * y1 ** 2)
    return x2, y2


def fit_cv(param, x1, y1):
    [k1, k2, k3, p1, p2] = param
    fx = 6.249257412785225370e+02
    fy = 6.325064978386543544e+02
    x1 /= fx
    y1 /= fy
    y1 *= -1
    r_2 = x1 ** 2 + y1 ** 2
    r_4 = r_2 ** 2
    r_6 = r_2*r_4
    x2 = x1 * (1 + k1 * r_2 + k2 * r_4+k3*r_6) + 2 * \
        p1 * x1 * y1 + p2 * (r_2 + 2 * x1 ** 2)
    y2 = y1 * (1 + k1 * r_2 + k2 * r_4+k3*r_6) + 2 * \
        p2 * x1 * y1 + p1 * (r_2 + 2 * y1 ** 2)
    x2 *= fx
    y2 *= fy
    y2 *= -1
    return x2, y2


def calc_distortion(param, points, logging=False):
    points1 = []
    points2 = []
    points3 = []
    points4 = []
    for point in points:
        if point[1] < 245 and point[0] > 315:
            points1.append(point)
        if point[1] < 245 and point[0] < 325:
            points2.append(point)
        if point[1] > 235 and point[0] < 325:
            points3.append(point)
        if point[1] > 235 and point[0] > 315:
            points4.append(point)
    isRotated = True
    columns1 = extract_points(points1, 0, rotated=isRotated)
    columns2 = extract_points(points2, 0, x_rev=True, rotated=isRotated)
    columns3 = extract_points(points3, 0, x_rev=True,
                              y_rev=True, rotated=isRotated)
    columns4 = extract_points(points4, 0, y_rev=True, rotated=isRotated)
    result = []
    # rotated
    for column in columns1:
        result.extend(column)
    for column in columns2:
        if column[0][0] == 0:
            result.extend(column[1:])
        else:
            result.extend(column)
    columns3.remove(columns3[0])
    for column in columns3:
        if column[0][0] == 0:
            result.extend(column[1:])
        else:
            result.extend(column)
    columns4.remove(columns4[0])
    for column in columns4:
        result.extend(column)
    # """
    result -= np.array([0, 0, 320, 240])
    result *= np.array([1, 1, 1, -1])
    calib_data = np.array(result)
    res = optimize.leastsq(fit, param, args=(calib_data.T))
    fit(res[0], calib_data.T)
    if logging:
        print(calib_data)
    find_points(res[0], calib_data, 'res')
    return res[0]


def find_points(param, calib_data, win_name):
    width = 640
    height = 480
    prev_point_map = np.zeros((480, 640))
    for i in range(len(calib_data)):
        prev_point_map[240-int(calib_data[i][3])][320 +
                                                  int(calib_data[i][2])] = i+1
    [_, k1, k2, _, k4, k5, _, p1, p2, _] = param
    """
    point_map = cv2.undistort(prev_point_map, np.array([[6.249257412785225370e+02, 0.000000000000000000e+00, 320],
                                                        [0.000000000000000000e+00, 6.325064978386543544e+02,
                                                         240],
                                                        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]),
                              np.array([-7.049532101626395653e-01, 3.287627385912758604e-01,
                                        2.619044002492451964e-02, -8.034101222118361987e-03, 8.844685025903741005e-02]))
    print(point_map.shape)
    """
    point_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x1 = j - width / 2
            y1 = height / 2 - i
            # x2, y2 = fit_cv([-7.049532101626395653e-01, 3.287627385912758604e-01, 8.844685025903741005e-02,
            #                 2.619044002492451964e-02, -8.034101222118361987e-03], x1, y1)
            # print(x2, y2)
            # x2, y2 = fit_old([-1.14864300e-06,  5.06219664e-14, -
            #                  4.55417377e-25, -1.64498475e-05,  6.69067565e-06], x1, y1)
            # x2, y2 = fit_current([1.05791597e-06,  5.26154073e-14, 3.41991153e-06,
            #                      3.27612688e-13, -4.30326545e-06, -4.60648477e-06], x1, y1)
            x2, y2 = fit_current([k1, k2, k4, k5, p1, p2], x1, y1)
            #x2, y2 = x1, y1
            ii = int(240-y2)
            jj = int(x2 + 320)
            if ii < 0 or ii >= 480:
                continue
            if jj < 0 or jj >= 640:
                continue
            point_map[i][j] = prev_point_map[ii][jj]
    # """
    points = []
    binary = cv2.inRange(point_map, 1, 10000)
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_image = np.zeros((height, width))
    new_calib_data = []
    for contour in contours:
        # print(contour[0])
        # print(point_map[contour[0][0][1]][contour[0][0][0]])
        (x, y), _ = cv2.minEnclosingCircle(contour)
        if (0 <= x and x < width and 0 <= y and y < height):
            points.append((x, y))
            color = point_map[contour[0][0][1]][contour[0][0][0]]
            index = int(color)-1
            res_image[int(y)][int(x)] = index + 1
            new_calib_data.append(
                [calib_data[index][0], calib_data[index][1], x - width / 2, height / 2 - y])
    new_calib_data = np.array(new_calib_data)
    calculate_variance(new_calib_data)
    cv2.imshow(win_name, res_image)
    cv2.imshow('repro', prev_point_map)
    print('show image')
    return points


def calculate_variance(calib_data):
    thres = 0.1
    dists = np.array([])
    print(calib_data)
    for i in range(len(calib_data)):
        # if calib_data[i][0] == 0 and calib_data[i][1] == 0:
        #    continue
        for j in range(i + 1, len(calib_data)):
            # if calib_data[j][0] == 0 and calib_data[j][1] == 0:
            #    continue
            dist = ((calib_data[i][0] - calib_data[j][0]) ** 2 +
                    (calib_data[i][1] - calib_data[j][1]) ** 2) ** 0.5
            dist2 = ((calib_data[i][2] - calib_data[j][2]) ** 2 +
                     (calib_data[i][3] - calib_data[j][3]) ** 2) ** 0.5
            if dist > 5-thres and dist < 5+thres:
                if dist2 < 15 or dist2 > 35:
                    pass
                    # print(calib_data[i], calib_data[j])
                dists = np.append(dists, dist2)
    mean = np.sum(dists) / len(dists)
    variance = 0
    for dist in dists:
        # print(dist, (dist-mean)**2)
                    # print(points[i], points[j])
        variance += (dist - mean) ** 2
    variance /= len(dists)
    print(len(dists))
    print(mean)
    print(variance)


if __name__ == '__main__':
    image = cv2.imread('use_points_Eval.png')
    #image[240][320] = [255, 255, 255]
    points = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j][0] == 255:
                points.append((j, i))

    param = [4, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    res = calc_distortion(param, points, logging=True)
    print(res)
    groundtruth = np.zeros((480, 640))
    for column in annotation_array(30, 30):
        for point in column:
            i = -int(point[1] * 4) + 240
            j = int(point[0] * 4) + 320
            if 0 <= i and i < 480 and 0 <= j and j < 640:
                groundtruth[i][j] = 255
    cv2.imshow('groundtruth', groundtruth)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
