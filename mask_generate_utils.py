import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def remove_small_components(mask, min_size=1000):
    """
    mask: sam model에서 inference 된 마스크
    
    return: min_size보다 작은 구성 요소가 제거된 마스크
    """
    labeled_mask = measure.label(mask)
    unique_labels = np.unique(labeled_mask)
    
    for label in unique_labels:
        if np.sum(labeled_mask == label) < min_size:
            mask[labeled_mask == label] = 0
            
    return mask


def mask_post_process(mask, kernel_size=7, min_size=1000):
    """
    mask: sam model에서 inference된 이미지

    return
    torch.tensor([eroded_mask_np]): 모폴로지 변환이 적용된 마스크,
    contour: 마스크 외곽선점들의 array

    """

    mask = remove_small_components(mask, min_size=min_size)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 모폴로지 변환 적용 (예: 팽창 후 침식)
    dilated_mask_np = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask_np = cv2.erode(dilated_mask_np, kernel, iterations=1)

    return eroded_mask_np