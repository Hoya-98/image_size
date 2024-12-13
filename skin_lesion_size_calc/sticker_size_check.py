# 이미지 읽기
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np

import imutils
import cv2
import matplotlib.pyplot as plt

import gradio as gr


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def blue_sticker_detect(image):
    # BGR에서 HSV로 변환 # gradio에서는 BGR
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 파란색의 HSV 범위 정의
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])


    # 범위 내의 색상만 추출하는 마스크 생성
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 범위 내의 색상만 추출하는 마스크 생성
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 결과 이미지를 생성
    res = cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(res)
    # 컨투어 찾기
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"find:{cnts}")

    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 10000: # 범위 크기 지정
            continue

        approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True) # 외곽선 근사 하기

        if len(approx) < 8:
            continue
        else:
            cv2.drawContours(image, [c], 0, (0, 255, 0), 5)
            
            orig = image.copy()
            box = cv2.minAreaRect(c) # contour 최소 bbox 크기 설정
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box) # 박스 좌표 순서 추출 좟아단, 우상단, 우하단, 좌하단
            box = np.array(box, dtype="int") # np.array int형으로 변경

            # box = perspective.order_points(box)
            cv2.drawContours(orig, [box], -1, (0, 255, 0), 2) # bbox 그리기
            
            # bbox 그리기
            for (x, y) in box: 
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


            # 좌상단, 우상단, 우하단, 좌하단 box 나누기
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr) # Top left, Top right의 중간값
            (blbrX, blbrY) = midpoint(bl, br) # Bottom Left, Bottom Right의 중간값

            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl) # Top left, bottom left의 중간값
            (trbrX, trbrY) = midpoint(tr, br) # Top Right, bottom right의 중간값
            
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)


            # compute the Euclidean distance between the midpoints
            # 거리 구하기
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            long_distance = dB if dA <= dB else dA # 장축 길이 구하기
            
            pixelsPerMetric = 1.6 / long_distance

            # compute the size of the object
            dimA = dA * pixelsPerMetric
            dimB = dB * pixelsPerMetric

            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}cm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 0, 0), 3)
            cv2.putText(orig, "{:.1f}cm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 0, 0), 3)
            # show the output image

            # plt.figure(figsize=(20, 10))
            # plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            # plt.axis('off')

    return orig, (dimA, dimB)


demo =  gr.Interface(
            blue_sticker_detect,
            inputs=["image"],
            outputs=[gr.Image(label="Input Image"), gr.Textbox(label="Sticker length")],
            title = "Pressure Ulcer Demo",
            description="Measuring a picture with blue sticker"
        )

        
if __name__=="__main__":
    demo.launch(share=True)
