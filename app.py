from flask import Flask, render_template, Response, request
import base64
from PIL import Image
import re
from io import BytesIO


# 파일 import
import sign, sign2

app = Flask(__name__)


@app.route('/') # 홈페이지
def index():
    return render_template('index.html')

@app.route('/education') # 학습 페이지 
def education():
    return render_template('education.html')


@app.route('/canvas_image', methods=('GET', 'POST')) # ajax로 0.5초마다 이미지 받아옴 // 이미지 전송, 응답 라우트
def canvas_image():
    
    # 클라이언트에서 요청이 있으면
    if request.method == "POST":
        
        # Base64로 넘어온 data를 img형태로 변환
        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
        
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        # 이미지 서버에 저장
        im.save('canvas.png')
        
        # 체크
        text = sign.image() # 골격 정보 체크 함수
        if text == None:
            text = ""
    return text

@app.route('/translation') # 번역 페이지
def translation():
    return render_template('translation.html')

@app.route('/canvas_image2', methods=('GET', 'POST')) # ajax로 0.1초마다 이미지 받아옴 // 이미지 전송, 응답 라우트
def canvas_image2():
    
    # 클라이언트에서 요청이 있으면
    if request.method == "POST":
        
        # Base64로 넘어온 data를 img형태로 변환
        image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
        
        im2 = Image.open(BytesIO(base64.b64decode(image_data)))
        # 이미지 서버에 저장
        im2.save('canvas.png')
        
        # 체크
        text = sign2.output_label() # 단어 정보 체크 함수, 이미지 객체
        if text == None:
            text = ""
    return text

@app.route('/test') # 시험 페이지 
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True) # 
