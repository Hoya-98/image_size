import os
import sys
import cv2
sys.path.append('./libs')
import argparse
import base64
from calibrate_colors import main
import warnings
warnings.filterwarnings('ignore')


jf_canvas_coordinate1 = (590, 794, 163, 98)
mm_per_pixel, img = main('./test.jpg', jf_canvas_coordinate1)

width = float(jf_canvas_coordinate1[2])
height = float(jf_canvas_coordinate1[3])
diagonal = (width**2 + height**2)**(1/2)

width_mm = float(int(mm_per_pixel * width*100)/100)
height_mm = float(int(mm_per_pixel * height*100)/100)
diagonal_mm = float(int(mm_per_pixel * diagonal*100)/100)
text = "width: " + str(width_mm) + "  \nheight: " + str(height_mm) + " \ndiagonal: " + str(diagonal_mm)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("example.png", img)
print(text)