# uvicorn SAM_api:app --host=0.0.0.0 --port=8000 --reload
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
import base64
import os
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
import shutil
from segment_anything import sam_model_registry
from mask_generate import mask_generate

#######################################################################################################################################
"""
import random
import sys
import pandas as pd
import ast # 문자열을 읽어 list로 바꿔주는 함수
from PIL import ExifTags
from skimage import measure
from PIL.ExifTags import TAGS
import tqdm
import json
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
from mask_generate_utils import midpoint, remove_small_components, mask_post_process
"""
#######################################################################################################################################

# SAM 모델의 체크포인트와 설정을 정의합니다.
sam_checkpoint = "/arthur/SAM/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda:1" if torch.cuda.is_available() else "cpu"  # CUDA를 사용할 수 있는지 확인 후 설정
# device = "cpu"  # 만약 GPU가 없을 경우 주석을 해제하고 CPU로 강제 설정
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.load_state_dict(torch.load("./sam_best.pth"))
sam.to(device=device)

#######################################################################################################################################

# FastAPI 앱을 초기화합니다.
app = FastAPI()
UNPROCESSED_FILE_DIRECTORY = Path("./unprocessed")  # 처리되지 않은 파일이 저장될 디렉터리
PROCESSED_FILE_DIRECTORY = Path("./processed")  # 처리된 파일이 저장될 디렉터리


# CORS 정책을 설정합니다. 모든 도메인에서의 요청을 허용하도록 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (특정 도메인으로 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello World"}  # 기본 경로에 접근할 경우 간단한 메시지를 반환


@app.get("/unprocessed_files")
async def unprocessed_files():
    files = os.listdir(path=UNPROCESSED_FILE_DIRECTORY)  # 처리되지 않은 파일 목록을 반환
    return JSONResponse(content=files)


@app.get("/processed_files")
async def processed_files():
    files = os.listdir(path=PROCESSED_FILE_DIRECTORY)  # 처리된 파일 목록을 반환
    return JSONResponse(content=files)


@app.get("/get_processed_files")
async def get_processed_files(folder_name: str):
    folder_path = os.path.join(PROCESSED_FILE_DIRECTORY, folder_name)
    if not os.path.exists(folder_path):  # 지정된 폴더가 없을 경우 404 에러 반환
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' does not exist.")
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]  # 이미지 파일 필터링
    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")  # 이미지를 Base64 형식으로 인코딩
            images.append({
                "filename": image_file,
                "image_base64": image_base64
            })
    
    return images


@app.post("/upload_file")
async def upload_file(files: list[UploadFile] = File(...), coordinates: str = Body(...)):
    print(coordinates)
    # JSON 문자열을 딕셔너리로 변환
    coord = eval(coordinates)

    x1 = coord['x1']
    y1 = coord['y1']
    x2 = coord['x2']
    y2 = coord['y2']
    
    for file in files:
        content = await file.read()
        
        # 파일 이름에 좌표를 포함하여 새 파일 이름을 구성
        new_filename = f"{x1}_{y1}_{x2}_{y2}_{file.filename}"
        file_path = os.path.join(UNPROCESSED_FILE_DIRECTORY, new_filename)
        
        # 새 파일 이름으로 파일 저장
        with open(file_path, "wb") as fp:
            fp.write(content)
        
        print(f"업로드한 파일: {file.filename}, 좌표: {x1}, {y1}, {x2}, {y2}")

    return {"message": "파일 업로드 완료"}


@app.post("/inference")
async def inference():
    # 처리되지 않은 디렉터리의 파일을 순회
    for filename in os.listdir(UNPROCESSED_FILE_DIRECTORY):
        if filename.lower().endswith(".jpg"):
            # 파일 이름에서 좌표를 추출
            coords = [float(coord) for coord in filename.split("_")[:4]]
            
            # 이미지 로드
            image_path = os.path.join(UNPROCESSED_FILE_DIRECTORY, filename)
            img = Image.open(image_path).convert("RGB")
            img = ImageOps.exif_transpose(img)  # 이미지의 회전 문제를 해결
            image = np.array(img)
            
            print(f"original image shape: {image.shape}")
            # 마스크를 생성 (실제 마스크 생성 함수로 교체)
            masks = mask_generate(image, coords, sam)  # 실제 마스크 생성 함수로 교체
            
            # 첫 4개의 좌표를 제외한 파일 이름으로 폴더 이름 생성
            folder_name = "_".join(filename.split("_")[4:]).rsplit(".", 1)[0]
            output_folder = os.path.join(PROCESSED_FILE_DIRECTORY, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # 원본 이미지를 처리된 폴더로 이동
            shutil.move(image_path, os.path.join(output_folder, filename))
            
            # 생성된 마스크를 이미지로 저장
            for i, mask in enumerate(masks, start=1):
                print(f"{i}th mask shape: {mask.shape}")
                if i <= 3:
                    mask_path = os.path.join(output_folder, f"{folder_name}_{i}_origin.png")
                    cv2.imwrite(mask_path, mask)
                else:
                    mask_path = os.path.join(output_folder, f"{folder_name}_{i-3}.png")
                    mask[mask > 0] = 1
                    cv2.imwrite(mask_path, mask * 255)
            
            print(f"Processed image: {filename}, Coords: {coords}")
    
    return {"message": "이미지 처리 완료"}