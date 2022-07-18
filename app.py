from flask import Flask, render_template, Response, request
import base64
from PIL import Image
import re
from io import BytesIO


# 파일 import
import sign

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
        sign.image() # 골격 정보 체크 함수
        
        
    return ('', 204) # 아무것도 리턴하지 않음




@app.route('/translation')
def translation():
    return render_template('translation.html')

@app.route('/test') # 학습 페이지 
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run()
