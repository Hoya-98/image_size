import random
import cv2
import torch
import numpy as np
import os
import time
from mask_generate_utils import *
from segment_anything import sam_model_registry, SamPredictor
#######################################################################################################################################
"""
import gradio as gr
from gradio_image_prompter import ImagePrompter
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
import sys
from skimage import measure
import pandas as pd
from PIL import Image
import ast # 문자열을 읽어 list로 바꿔주는 함수
import tqdm
"""
#######################################################################################################################################

sam_checkpoint = "./SAM/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.load_state_dict(torch.load("./sam_best.pth"))
sam.to(device=device)

#######################################################################################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def mask_generate(img, text):
    x1, y1, x2, y2 = map(int, text)

    predictor = SamPredictor(sam)
    predictor.set_image(img)

    box = np.array([x1, y1, x2, y2]) # np.array(x1, y1, x2, y2)
    # input_point = np.array([[(x2+x1)*0.5, (y2+y1)*0.5]]) # np.array([[px1, py1], [px2, py2], [px3, py3]]]) 
    # input_label = np.array([0]) # np.array([0])

    input_point = None
    input_label = None

    t1 = time.time()
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
            box = box,
            multimask_output = True, 
        )
    t2 = time.time()
    print(f"걸린시간은 inference: {t2-t1}")

    mask_with_image_list = []
    mask_image_list = []

    t3 = time.time()

    color = np.array([30, 144, 255], dtype=np.float32) / 255
    for mask, score in zip(masks, scores):
        # mask = mask_post_process(mask.astype(np.uint8))
        # mask = remove_small_components(mask)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        mask_with_image = cv2.addWeighted(img.astype(np.uint8), 0.75, mask_image.astype(np.uint8)*255, 0.5, 0)
        mask_with_image = cv2.rectangle(mask_with_image, (x1, y1), (x2, y2), (0,0,0), 3)
        mask_with_image_list.append(mask_with_image)
        mask_image_list.append(mask_image)

    t4 = time.time()
    print(f"걸린시간 mask result: {t4-t3}")

    return mask_with_image_list[0], mask_with_image_list[1], mask_with_image_list[2], mask_image_list[0], mask_image_list[1], mask_image_list[2]



def preprocess(output):
    points = output['points'][0]
    x1 = points[0]
    x2 = points[3]
    y1 = points[1]
    y2 = points[4]
    path = output['path']

    mask_with_image1, mask_with_image2, mask_with_image3, mask_image1, mask_image2, mask_image3 = mask_generate(output['image'], [x1, y1, x2, y2])
    
    if mask_with_image1.shape != mask_image1.shape:
        print("Error")
        
    
    t1 = time.time()
    mask_1_path = os.path.join("./mask_list", path.replace(".jpg", "_1.png"))
    mask_2_path = os.path.join("./mask_list", path.replace(".jpg", "_2.png"))
    mask_3_path = os.path.join("./mask_list", path.replace(".jpg", "_3.png"))
    
    mask_image1[mask_image1 > 0] = 1
    mask_image2[mask_image2 > 0] = 1
    mask_image3[mask_image3 > 0] = 1

    cv2.imwrite(mask_1_path, mask_image1*255)
    cv2.imwrite(mask_2_path, mask_image2*255)
    cv2.imwrite(mask_3_path, mask_image3*255)
    print(f"time {time.time()-t1}")
    
    return mask_with_image1, mask_1_path, mask_with_image2, mask_2_path, mask_with_image3, mask_3_path

"""
demo = gr.Interface(
    preprocess,
    ImagePrompter(show_label=True),
    [gr.Image(label="Mask1_with_image", format="png"), gr.Image(label="Mask1", format="png"),
     gr.Image(label="Mask2_with_image", format="png"), gr.Image(label="Mask2", format="png"),
     gr.Image(label="Mask3_with_image", format="png"), gr.Image(label="Mask3", format="png")],
    title="Burn SAM",
    description="Drawing a BBox and then Select Mask",
)



if __name__=="__main__":
    # demo.launch(debug=True, share=True, server_name="7.acryl.ai", server_port=30002)
    demo.launch(share=True, server_name="0.0.0.0", server_port=8081)
"""