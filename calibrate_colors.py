import os,sys
from os.path import join as pjoin
import glob
from PIL import Image, ImageOps
import numpy as np
import pandas as apd
import cv2
# from libs import colour
from color import adjust_colors
from measure import detect_aruco, check_marker_type, compute_burn_sizes, draw_region_aruco
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = pjoin('data')
OUTPUT_PATH = pjoin('output')
XLS_NAME = 'bbox'
CCM_METHOD = 1  # method 1: cheung 2004, 2: Finlayson 2015, 3: Vandermonde



def main(img_name, xyxy_coordinates) -> None:
    """
    arguments:
        img_name: 입력하고자 하는 이미지 경로
        xyxy_coordinates: 찾고자하는 bbox

    return:
        len_per_pixel: 픽셀의 길이
        img: draw 박스가 그려진 결과 이미지
    
    """
    img_path = pjoin('')

    ori_img = Image.open(img_name).convert("RGB")
    ori_img = ImageOps.exif_transpose(ori_img)
    ori_img = np.array(ori_img)
    # ori_img = cv2.cvtColor(cv2.imread(pjoin(img_path, img_name)), cv2.COLOR_BGR2RGB)
    print(f"Image Shape: {ori_img.shape}")
    (corners, ids, rejected) = detect_aruco(ori_img, img_name)
    
    if not rejected:
        print("cannot find marker for img " + str(img_name))
        return 1
    else:
        print("yes marker: " + str(len(corners)))
        print(f"Input Image name: {img_name}")
    
    #(selected_corners, selected_ids, color_marker_type) = check_marker_type(corners, ids, img_name)
    try:
        (selected_corners, selected_ids, color_marker_type) = check_marker_type(corners, ids, img_name)
        len_per_pixel, color_marker_img = compute_burn_sizes(ori_img, selected_corners, selected_ids, color_marker_type)
    except:
        print("not enough marker detected")
        return 1

    print("len_per_pixel " + str(len_per_pixel))
    img = adjust_colors(ori_img, color_marker_img, color_marker_type, CCM_METHOD)
    img = draw_region_aruco(img, corners, ids)

    # in order to calculate size(temporary)
    # sqrt(23^2 + 638^2) * 0.150162~= 95.8mm  # 0.15016254832107795
    # x1 = 100
    # x2 = 100
    # y1 = 200
    # y2 = 200
    # # x1, y1 = 565, 831
    # x2, y2 = 552, 1178
    # sqrt(13^2 + 347^2) * 0.153647 ~= 53.3mm  # 0.15364752172400575

    x1, y1, w, h = map(int, xyxy_coordinates)

    x2 = x1 + w
    y2 = y1 + h

    len_width = abs(x2 - x1) * len_per_pixel
    len_height = abs(y2 - y1) * len_per_pixel

    point_color = (0, 0, 255)
    text_color = (0, 0, 0)

    # img = cv2.line(img, (x1, y1), (x1, y1), color=point_color, thickness=5)
    # img = cv2.line(img, (x2, y2), (x2, y2), color=point_color, thickness=5)

    # cv2.rectangle(img, (x1, y1), (x2, y2), color=line_color)

    img = cv2.putText(
        img, f'Width: {len_width:.1f}mm', (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2
    )

    img = cv2.putText(
        img, f'Height: {len_height:.1f}mm', (x1, y1 - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2
    )

    save_name = f'after_aruco_{img_name}'
    print(f"Output Image name: {save_name}")
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,255,255), thickness=3)
    cv2.imwrite(pjoin(img_path, save_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return len_per_pixel, img
    

if __name__ == '__main__':
    img_name = 'test.jpg'
    xywh_coordinates = (590, 794, 163, 98)
    main(img_name, xywh_coordinates)
