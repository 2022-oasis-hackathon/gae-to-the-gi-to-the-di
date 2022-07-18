# 골격 체크 파일


import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

gesture = {
    0:'너는', 1:'이름', 2:'뭐야?', 3:'어디', 4:'살다', 5:'좋다',
}
#rps_gesture = {0:'rock', 5:'paper', 9:'scissors', }


file = np.genfromtxt('data/gesture_data.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)










# 영상 input
def gen(video):
    
    while True:
        
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            
            
                while video.isOpened():
                    success, image = video.read()
                    image = cv2.flip(image, 1) #inversed frame
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue

                    
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    results2 = hands.process(image)
                    
                    
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    
                    
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                
                    ## 좌표 정보 처리
                    landmarks = results.pose_landmarks.landmark
                    print(landmarks[12].x ,landmarks[11].x)
        
        
        
        
        
                    if results2.multi_hand_landmarks:
                        for hand_landmarks in results2.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        

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
                                print(gesture[idx])
                        
                        
                        
                        
                        
                        
                        
                        
                        # 좌표 정보 처리
                        landmarks2 = results2.multi_hand_landmarks
                        if len(landmarks2) > 0:
                            print(landmarks2[0].landmark[0].x)
        

                    # 웹캠 이미지 전송
                    ret, jpeg = cv2.imencode('.jpg', image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                
                
# image input    
def image():

    IMAGE_FILES = ['canvas.png']
    
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
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
                
                if not results.pose_landmarks:
                    continue
                
                landmarks = results.pose_landmarks.landmark
                print(landmarks[12].x ,landmarks[11].x)
                    
                
                
                
                if results2.multi_hand_landmarks:
                    
                    
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
                                print(gesture[idx])
                    
                    
                    
                    
                    
                    landmarks2 = results2.multi_hand_landmarks
                    if len(landmarks2) > 0:
                        print(landmarks2[0].landmark[0].x)
            
    

            
                   

    
           
            
