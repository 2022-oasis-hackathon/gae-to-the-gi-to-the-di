from flask import Flask, render_template, Response, request
import mediapipe as mp
import base64
from PIL import Image
import re
from io import BytesIO
import cv2

# 파일 import
import sign



app = Flask(__name__)

################################################## test 시 주석 해제 -> 테스트용 링크 클릭
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# video = cv2.VideoCapture(1)
##################################################

@app.route('/') # 홈페이지
def index():
    return render_template('index.html')

@app.route('/education') # 웹에서 촬영-> 서버로 전송-> 포즈 인식-> 정보 처리-> 결과 전송 # 현재 골격 표시는 안(못) 하고 있음
def edu():
    return render_template('edu.html')


@app.route('/canvas_image', methods=('GET', 'POST')) # ajax로 0.5초마다 이미지 받아옴
def canvas_image():
    
    
    if request.method == "POST":
        
        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
        
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        
        im.save('canvas.png')
        
        sign.image() # 골격 정보 체크 함수
        
        
    return ('', 204) # 아무것도 리턴하지 않음




@app.route('/translation')
def translation():
    return render_template('translation.html')



@app.route('/test') # 서버에서 촬영, 인식, 처리, 결과 전송
def video_feed():
		
    global video

    return Response(sign.gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run()
