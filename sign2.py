# 골격 체크 파일
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

model = load_model('dynamic_model/models/model.h5') # 모델 불러오기

# mediapipe 변수
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#cap = cv2.VideoCapture(0)

actions = [ '안녕하세요', '여러분', '발표', '시작', '오늘', '하루', '어떻게', '보내다']
#actions = ['everyone','hello','presentation','start','how','today', 'day','spend']

seq = []
seq_length = 30
action_seq = []

def output_label():
    #while cap.isOpened():
        #ret,img = cap.read()
    IMAGE_FILES = ['canvas.png']

    img = cv2.imread('canvas.png')
    if img is None:
        return
    
    #for idx, file in enumerate(IMAGE_FILES):
    #img = cv2.imread('canvas.png')
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            
            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree
            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            
            if len(seq) < seq_length: 
                continue
            
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)
            
            if len(action_seq) < 3:
                continue

            this_action = '?'
            
            flag = True
            for i in action_seq[-3:]:
                if action != i:
                    flag = False
                    break
            if flag: this_action = action 
            if this_action != '?' : return this_action