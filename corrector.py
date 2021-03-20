import sys
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


def extract_positions(points, annotation_begin, x_rev=False, y_rev=False, rotated=False):
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
    return columns


def fit(param, x):
    [d, k1, k2, k3, k4, p1, p2] = param
    x1 = d * x[0]
    y1 = d * x[1]
    x2 = x[2]
    y2 = x[3]
    r_2 = np.square(x1) + np.square(y1)
    r_4 = np.square(r_2)
    x_delta = np.square(x1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) +
                        2 * p1 * x1 * y1 + p2 * (r_2 + 2 * np.square(x1)) - x2)
    y_delta = np.square(y1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) +
                        2 * p2 * x1 * y1 + p1 * (r_2 + 2 * np.square(y1)) - y2)
    return x_delta + y_delta


def correct(param, x):
    [k1, k2, k3, k4, p1, p2] = param
    r_2 = x[0] ** 2 + x[1] ** 2
    r_4 = r_2 ** 2
    x2 = x[0] * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * \
        p1 * x[0] * x[1] + p2 * (r_2 + 2 * x[0] ** 2)
    y2 = x[1] * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * \
        p2 * x[0] * x[1] + p1 * (r_2 + 2 * x[1] ** 2)
    return x2, y2


def calc_distortion(param, points, logging=False):
    # 1-4象限に分割してキャリブレーションデータを取得
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
    isRotated = False
    columns1 = extract_positions(points1, 0, rotated=isRotated)
    columns2 = extract_positions(points2, 0, x_rev=True, rotated=isRotated)
    columns3 = extract_positions(points3, 0, x_rev=True,
                                 y_rev=True, rotated=isRotated)
    columns4 = extract_positions(points4, 0, y_rev=True, rotated=isRotated)
    result = []

    for column in columns1:
        result.extend(column)
    columns2.remove(columns2[0])
    for column in columns2:
        result.extend(column)
    columns3.remove(columns3[0])
    for column in columns3:
        if column[0][1] == 0:
            result.extend(column[1:])
        else:
            result.extend(column)
    for column in columns4:
        if column[0][1] == 0:
            result.extend(column[1:])
        else:
            result.extend(column)
    # 原点は除去
    result -= np.array([0, 0, 320, 240])
    # 実データのy座標については反転
    result *= np.array([1, 1, 1, -1])
    calib_data = np.array(result)

    res = optimize.leastsq(fit, param, args=(calib_data.T))
    if logging:
        print(calib_data)
    find_points(res[0], calib_data, 'Correction result')
    return res[0]


def find_points(param, calib_data, win_name):
    width = 1500
    height = 1000
    prev_point_map = np.zeros((480, 640))
    for i in range(len(calib_data)):
        prev_point_map[240-int(calib_data[i][3])][320 +
                                                  int(calib_data[i][2])] = i+1
    [d, k1, k2, k3, k4, p1, p2] = param

    point_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x1 = j - width / 2
            y1 = height / 2 - i
            x2, y2 = correct([k1, k2, k3, k4, p1, p2], [x1, y1])
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
    contours, _ = cv2.findContours(
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
    return points


def calculate_variance(calib_data):
    thres = 0.1
    dists = np.array([])
    for i in range(len(calib_data)):
        for j in range(i + 1, len(calib_data)):
            dist = ((calib_data[i][0] - calib_data[j][0]) ** 2 +
                    (calib_data[i][1] - calib_data[j][1]) ** 2) ** 0.5
            dist2 = ((calib_data[i][2] - calib_data[j][2]) ** 2 +
                     (calib_data[i][3] - calib_data[j][3]) ** 2) ** 0.5
            if dist > 5-thres and dist < 5+thres:
                if dist2 < 15 or dist2 > 35:
                    pass
                dists = np.append(dists, dist2)
    mean = np.sum(dists) / len(dists)
    variance = 0
    for dist in dists:
        variance += (dist - mean) ** 2
    variance /= len(dists)
    print('Evaluation done')
    print(f'Edges : {len(dists)}')
    print(f'Mean : {mean}')
    print(f'Variance : {variance}')


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print('No image file is specified!')
        exit()

    image = cv2.imread(args[1])
    if image is None:
        print('Invalid image format!')
        exit()

    points = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j][0] == 255:
                points.append((j, i))

    param = [4, 0, 0, 0, 0, 0, 0]
    res = calc_distortion(param, points, logging=True)
    print(f'Calibration params [ k1, k2, k3, k4, p1, p2 ] = {res[1:]}')

    groundtruth = np.zeros((480, 640))
    for column in annotation_array(30, 30):
        for point in column:
            i = -int(point[1] * 4) + 240
            j = int(point[0] * 4) + 320
            if 0 <= i and i < 480 and 0 <= j and j < 640:
                groundtruth[i][j] = 255
                groundtruth[479-i][j] = 255
                groundtruth[i][639-j] = 255
                groundtruth[479-i][639-j] = 255
    cv2.imshow('groundtruth', groundtruth)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished")
