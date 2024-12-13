import sys
import random
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import ast # 문자열을 읽어 list로 바꿔주는 함수
import tqdm
import os
from skimage import measure
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from make_mask_utils import *

from segment_anything import sam_model_registry, SamPredictor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(image):

    set_seed(42)

    sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # sam.load_state_dict(torch.load("./sam_best.pth"))

    predictor = SamPredictor(sam)
    predictor.set_image(orig)


    box = np.array([x1, y1, x2, y2]) # np.array(x1, y1, x2, y2)
    input_point = np.array([[(x2+x1)//2, (y2+y1)//2]]) # np.array([[px1, py1], [px2, py2], [px3, py3]]]) 
    input_label = np.array([0]) # np.array([0])

    input_point = None
    input_label = None

    masks, scores, logits = predictor.predict(
        point_coords = input_point,
        point_labels = input_label,
        box = box,
        multimask_output = True, 
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(orig)
        show_mask(mask, plt.gca())

    if input_point is not None:
        show_points(input_point, input_label, plt.gca())

    if box is not None:
        show_box(box, plt.gca())
        
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(orig)
    plt.axis('off')