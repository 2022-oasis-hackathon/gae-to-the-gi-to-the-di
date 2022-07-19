# 골격 체크 파일

import cv2
import mediapipe as mp
import numpy as np

# 파일 import
import sign_dic


# mediapipe 변수
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# 제스처 번호
# gesture = {
#     0:'너는', 1:'이름', 2:'뭐야?', 3:'어디', 4:'살다', 5:'좋다',
# }
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
#rps_gesture = {0:'rock', 5:'paper', 9:'scissors', }


# 제스처 모델 import
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)



# 단어 사전
array = {
    0 : '네',
    1 : '아니오',
    2 : '싫어',
    3 : '괜찮아',
    4 : '감사합니다',
    5 : '미안합니다',
    6 : '좋다',
    7 : '너는',
    8 : '이름',
    9 : '무엇',
    10 : '어디',
    11 : '살다',
    12 : '음식,먹다',
    13 : '어제',
    14 : '놀다',
    15 : '나',
    16 : '같이',
    17 : '가자'
}

# 현재 동작 번호
checklist = [0,0,0,0,0 ,0,0,0,0,0, 0,0,0,0,0, 0,0,0,0]

# 무동작 시 체크 변수
none_count = 0
         
                
# image input    
def image():
    
    global checklist, none_count
    
    IMAGE_FILES = ['canvas.png']
    
    # 이미지에서 골격 체크
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        

            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                
                
                
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                results2 = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # 사람이 없으면? 함수 종료
                if not results.pose_landmarks:
                    checklist = [0 for i in range(len(checklist))]  
                    return 
                
                # 포즈 랜드마크 저장
                landmarks = results.pose_landmarks.landmark
                
                    
                # 손이 이미지에 있으면
                if results2.multi_hand_landmarks:
                    # 제스처 담는 변수
                    rps_result = []
                    #################################################### 제스처 인식 줄
                    for res in results2.multi_hand_landmarks:
                        
                            joint = np.zeros((21, 3))
                            for j, lm in enumerate(res.landmark):
                                joint[j] = [lm.x, lm.y, lm.z]

                            # Compute angles between joints
                            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                            v = v2 - v1 # [20,3]
                            # Normalize v
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            # Get angle using arcos of dot product
                            angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                            angle = np.degrees(angle) # Convert radian to degree

                            # Inference gesture
                            data = np.array([angle], dtype=np.float32)
                            ret, results, neighbours, dist = knn.findNearest(data, 3)
                            idx = int(results[0][0])
                        
                            if idx in gesture.keys():
                                rps_result.append(gesture[idx])
                        
                    print(rps_result)
                    ##################################################### 여기까지 제스처 인식
                    
                    # 손 랜드마크 저장
                    landmarks2 = results2.multi_hand_landmarks
                    
            
                    # 현재 동작이 수어 동작인지 체크 함수
                    result = sign_dic.check(landmarks, landmarks2, rps_result)
                    
                    
                    if result != None:
                        checklist[result] += 1
                    else:
                        none_count += 1
                    print(checklist)
                    
                    ########################### 단어 모음 //  이후 웹으로 정보 넘길 땐 return array[i] 하면 됨
                    # 예
                    if checklist[0] > 0:
                        print(array[0])
                        checklist = [0 for i in range(len(checklist))]
                        return array[0]
                    # 아니오
                    elif checklist[1] > 0:
                        if checklist[2] > 0:
                            print(array[1])
                            checklist = [0 for i in range(len(checklist))]
                            return array[1]
                        # 살다
                        elif checklist[12] > 0:
                            print(array[11])
                            checklist = [0 for i in range(len(checklist))]
                            return array[11]
                    # 싫어
                    elif checklist[3] > 0:
                        print(array[2])
                        checklist = [0 for i in range(len(checklist))]
                        return array[2]
                    # 괜찮아
                    elif checklist[4] > 0:
                        print(array[3])
                        checklist = [0 for i in range(len(checklist))]
                        return array[3]
                    #미안 감사
                    elif checklist[5] > 0:
                        if checklist[6] > 0:
                            print(array[5]) # 미안합니다
                            checklist = [0 for i in range(len(checklist))]
                            return array[5]
                        else:
                            print(array[4]) # 감사합니다
                            checklist = [0 for i in range(len(checklist))]
                            return array[4]
                    
                    # 좋다
                    elif checklist[7] > 0:
                        print(array[6])
                        checklist = [0 for i in range(len(checklist))]
                        return array[6]
                    
                    # 너
                    elif checklist[8] > 0:
                        print(array[7])
                        checklist = [0 for i in range(len(checklist))]
                        return array[7]
                        
                    # 이름
                    elif checklist[9] > 0:
                        print(array[8])
                        checklist = [0 for i in range(len(checklist))]
                        return array[8]
                        
                    # 무엇
                    elif checklist[10] > 0:
                        print(array[9])
                        checklist = [0 for i in range(len(checklist))]
                        return array[9]
                    
                    # 어디
                    elif checklist[11] > 0:
                        print(array[10])
                        checklist = [0 for i in range(len(checklist))]
                        return array[10]
                    
                    # 먹다
                    elif checklist[13] > 0:
                        print(array[12])
                        checklist = [0 for i in range(len(checklist))]
                        return array[12]
                        
                    # 어제
                    elif checklist[14] > 0:
                        print(array[13])
                        checklist = [0 for i in range(len(checklist))]    
                        return array[13]
                        
                    # 놀다
                    elif checklist[15] > 0:
                        print(array[14])
                        checklist = [0 for i in range(len(checklist))]  
                        return array[14]
                    
                    # 나
                    elif checklist[16] > 0:
                        print(array[15])
                        checklist = [0 for i in range(len(checklist))]  
                        return array[15]    
                        
                    # 같이
                    elif checklist[17] > 0:
                        print(array[16])
                        checklist = [0 for i in range(len(checklist))]
                        return array[16]    
                        
                    # 가자
                    elif checklist[18] > 0:
                        print(array[17])
                        checklist = [0 for i in range(len(checklist))]  
                        return array[17]   
                        
                # 아무 행동도 없을 시 초기화
                elif none_count > 5:
                    checklist = [0 for i in range(len(checklist))]
                    none_count = 0
           
            
