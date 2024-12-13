import subprocess
#from subprocess import STDOUT, check_call
import os
#check_call(['pip', 'intall', '-y', '-U', 'numpy==1.24'], stdout=open(os.devnull,'wb'), stderr=STDOUT)
# subprocess.run(['pip','install', 'numpy==1.23']) #, shell=True, stdin=None, stdout=None, stderr=None,
# subprocess.run(['pip','install', 'opencv-python==4.6.0.66'])
# subprocess.run(['pip','install', 'opencv-contrib-python==4.7.0.72'])
#subprocess.run(['pip','install', 'opencv-python==4.6.0'])



"""
시스템 정보
아래 스크립트 삭제시 JF deploy 실행이 안됩니다.
#JF_DEPLOYMENT_INPUT_DATA_INFO_START
{
    "deployment_input_data_form_list": [
        {
            "method": "POST",
            "location": "file",
            "api_key": "image",
            "value_type": ".jpg",
            "category": "canvas-image",
            "category_description": ""
        },
        {
            "method": "POST",
            "location": "form",
            "api_key": "coordinate",
            "value_type": "string",
            "category": "canvas-coordinate",
            "category_description": "box"
        }
    ]
}
#JF_DEPLOYMENT_INPUT_DATA_INFO_END
"""
import sys
sys.path.append('/addlib')
from deployment_api_deco import api_monitor
from flask import Flask, request, jsonify
from flask.views import MethodView
from flask_cors import CORS
import argparse
import base64
from calibrate_colors import main

"""
배포 실행 명령어 관련 자동생성 영역
"""

"""
사용자 추가 영역
"""


app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})

class run_api(MethodView):
    def __init__(self):
        pass

    @api_monitor()
    def get(self):
        return "JF DEPLOYMENT RUNNING"

    @api_monitor()
    def post(self):
        """
        TEST API 받아오는 부분 자동 생성 영역
        """
        jf_canvas_coordinate1 = request.form['coordinate'].split(",")
        jf_canvas_image1_files = request.files.getlist('image')
        jf_canvas_image1 = request.files['image']
        
        jf_canvas_image1.save('./test.jpg')
        
        """
        사용자 영역
        # 필요한 전처리
        # 배포 테스트에 필요한 처리 등 
        """
        mm_per_pixel = main('./test.jpg')
        text2 = str(jf_canvas_coordinate1)# + " , " + str(jf_canvas_coordinate1[1])
        width = float(jf_canvas_coordinate1[2]) - float(jf_canvas_coordinate1[0])
        height = float(jf_canvas_coordinate1[3]) - float(jf_canvas_coordinate1[1])
        diagonal = (width**2 + height**2)**(1/2)

        width_mm = float(int(mm_per_pixel * width*100)/100)
        height_mm = float(int(mm_per_pixel * height*100)/100)
        diagonal_mm = float(int(mm_per_pixel * diagonal*100)/100)
        text = "width: " + str(width_mm) + "  \nheight: " + str(height_mm) + " \ndiagonal: " + str(diagonal_mm)

        """
        Output 자동 생성 영역 (OUTPUT Category)
        """
        output = {
            "text": [
                {
                    "@TITLE": text,
                    "@box": text2,
                    "@mm_per_pixel": str(mm_per_pixel)
                }
            ]
        }
        
        return jsonify(output)

app.add_url_rule("/", view_func=run_api.as_view("run_api"))
if __name__ == "__main__":
    """
    모델 로드를 권장하는 위치
    사용자 영역
    """
    app.run('0.0.0.0',8555,threaded=True, debug=True)