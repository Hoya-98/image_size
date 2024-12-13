from math import sqrt, pow
from typing import Iterable
import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
import cv2


def get_distance(vector: Iterable) -> float:
    # return sqrt(vector[0] ** 2 + vector[1] ** 2)
    return sqrt(sum([x ** 2 for x in vector]))


def four_point_transform(image: NDArray, pts: NDArray) -> NDArray:
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = int(norm(br - bl))
    width_b = int(norm(tr - tl))
    max_width = max(width_a, width_b)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = int(norm(tr - br))
    height_b = int(norm(tl - bl))
    max_height = max(height_a, height_b)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(pts, destination)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    # return the warped image
    return warped


def detect_aruco(img, img_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, rejected  = detector.detectMarkers(img)
    #corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

    if ids is None:
        raise ValueError(f"can't detect any aruco marker in {img_name}")

    return corners, ids, rejected


def check_marker_type(corners, ids, img_name):

    def _check_id_conditions(k: int) -> bool:
        return ((4 * k + 1 <= ids) & (ids <= 4 * (k + 1))).sum() >= 3
        # return (4 * k + 1 <= ids <= 4 * (k + 1)).sum() >= 3

    squeeze_positions = {1: np.where((ids > 4))}
    squeeze_positions.update({k: np.where((ids > 4 * k) | (ids < 4 * (k - 1) + 1)) for k in range(2, 5)})

    if _check_id_conditions(0):
        marker_type = 1
    elif _check_id_conditions(1):
        marker_type = 2
    elif _check_id_conditions(2):
        marker_type = 3
    elif _check_id_conditions(3):
        marker_type = 4
    else:
        raise ValueError(f'the number of detected aruco marker is low to calculate in {img_name}')

    n = np.squeeze(squeeze_positions[marker_type])
    ids = np.delete(ids, n[0])
    corners = np.delete(corners, n[0], axis = 0)

    return corners, ids, marker_type


def compute_burn_sizes(img, corners, ids, marker_type):

    def _find_point(marker_pos, pos: int):
        n = np.squeeze(np.where(ids == marker_pos))
        return np.squeeze(corners[n])[pos] if n.size != 0 else np.array([])

    def _define_fourth_point(t_l, t_r, b_r, b_l):
        # top_left, top_right, bot_right, bot_left
        # Note that top_left + bot_right = top_right + bottom_left
        if not t_l.size or not b_r.size:
            fourth_point = t_r + b_l
            if t_l.size:
                fourth_point -= t_l
                b_r = fourth_point
            if b_r.size:
                fourth_point -= b_r
                t_l = fourth_point

        elif not t_r.size or not b_l.size:
            fourth_point = t_l + b_r
            if t_r.size:
                fourth_point -= t_r
                b_l = fourth_point
            if b_l.size:
                fourth_point -= b_l
                t_r = fourth_point

        else:
            raise ValueError

        return t_l, t_r, b_r, b_l

    marker_dict = {
        'bw_height': {1: 19, 2: 34, 3: 49, 4: 64},
        'bw_width': {1: 33, 2: 58, 3: 83, 4: 108},
        'marker_length': {k: 5 * k for k in range(1, 5)}
    }

    marker_bw_height = marker_dict['bw_height'][marker_type]
    marker_bw_width = marker_dict['bw_width'][marker_type]
    # marker_length = marker_dict['marker_length'][marker_type]

    ids = ids.flatten()
    new_corners = [corner.reshape(4, 2) for corner in corners]

    # if len(new_corners) in (3, 4):  # aruco marker가 3개 이상으로 탐지 되었을때
        # position of aruco marker in color marker paper: top-left, top-right, bottom-right, and bottom-left [1,2,4,3]
        # if marker type is 2, position of aruco marker in color marker paper: top-left, top-right, bottom-right, and bottom-left [5,6,8,7]
        # if marker type is 3, ...
        # points in aruco marker: top-left, top-right, bottom-right, and bottom-left [0,1,2,3] 
    top_left = _find_point(1 + 4 * (marker_type - 1), 0)
    top_right = _find_point(2 + 4 * (marker_type - 1), 1)
    bottom_right = _find_point(4 + 4 * (marker_type - 1), 2)
    bottom_left = _find_point(3 + 4 * (marker_type - 1), 3)

    if len(new_corners) == 3:
        # estimate 4th point
        # if top_left.size == 0: top_left = top_right + bottom_left - bottom_right
        # if top_right.size == 0: top_right = top_left + bottom_right - bottom_left
        # if bottom_right.size == 0: bottom_right = top_right + bottom_left - top_left
        # if bottom_left.size == 0: bottom_left = top_left + bottom_right - top_right
        top_left, top_right, bottom_right, bottom_left = _define_fourth_point(
            top_left, top_right, bottom_right, bottom_left
        )
        print(str(top_left) + str(top_right) + str(bottom_right) + str(bottom_left) )
        len_per_pixel = sqrt(pow(marker_bw_width, 2) + pow(marker_bw_height, 2)) / max(get_distance(top_left - bottom_right), get_distance(top_right - bottom_left))
    elif len(new_corners) == 2:
        
        top_left, top_right, bottom_right, bottom_left = _define_fourth_point(
            top_left, top_right, bottom_right, bottom_left
        )

        try:
            len_per_pixel = marker_bw_width / max(get_distance(top_left - top_right), get_distance(bottom_left - bottom_right))
        except:
            len_per_pixel = marker_bw_height / max(get_distance(top_left - bottom_left), get_distance(top_right, bottom_right))
    elif len(new_corners)==4:
        
        print(str(top_left) + str(top_right) + str(bottom_right) + str(bottom_left) )
        len_per_pixel = sqrt(pow(marker_bw_width, 2) + pow(marker_bw_height, 2)) / max(get_distance(top_left - bottom_right), get_distance(top_right - bottom_left))
 

    # mm per pixel
    points = np.array([top_left, top_right, bottom_right, bottom_left])
    color_marker_img = four_point_transform(img, points)

    return len_per_pixel, color_marker_img


def draw_region_aruco(img, corners, ids):
    img = img.copy()

    for (marker_corner, marker_id) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = marker_corner.reshape(4, 2)
        (top_left, top_right, bottom_right, bottom_left) = corners

        # convert each of the (x, y)-coordinate pairs to integers
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))

        # img = np.zeros((4000, 4000, 3), np.uint8)
        # draw the bounding box of the ArUCo detection
        cv2.line(img, top_left, top_right, (0, 255, 0), 2)
        cv2.line(img, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(img, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(img, bottom_left, top_left, (0, 255, 0), 2)

        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
        cv2.circle(img, (center_x, center_y), 4, (0, 0, 255), -1)

        # draw the ArUco marker ID on the image
        cv2.putText(
            img, str(marker_id), (top_left[0], top_left[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

    return img