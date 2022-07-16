# 골격 체크 파일


import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands



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
                        
                        # 좌표 정보 처리
                        landmarks2 = results2.multi_hand_landmarks
                        if len(landmarks2) > 1:
                            print('2개')
        

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
                
                if not results.pose_landmarks:
                    continue
                
                landmarks = results.pose_landmarks.landmark
                print(landmarks[12].x ,landmarks[11].x)
                    
                print('Handedness:', results2.multi_handedness)
                
                if not results2.multi_hand_landmarks:
                    continue
            
                landmarks2 = results2.multi_hand_landmarks
                print(landmarks2) 
            
    
        
            
                   

    
           
            
