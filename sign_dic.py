
# pose : landmark 
# hand : landmark2


array = {
    0 : '네',
    1 : '아니오1',
    2 : '아니오2',
    3 : '싫어',
    4 : '괜찮아',
    5 : '감사합니다',
    6 : '미안합니다',
    7 : '좋다'
}


def check(landmarks, landmarks2, rps_result):
    
    # 네, 응, 맞아
    if len(landmarks2) == 1: # 손이 있을 때
        if len(rps_result) > 0:
            if landmarks[10].x < landmarks2[0].landmark[12].x and landmarks2[0].landmark[12].x < landmarks[9].x: # 12번이 9번 10번 사이
                
                if landmarks[12].x < landmarks2[0].landmark[4].x and landmarks2[0].landmark[4].x < landmarks2[0].landmark[5].x and landmarks2[0].landmark[5].x < landmarks[11].x:
                    if abs(landmarks[9].y - landmarks2[0].landmark[12].y)  < 0.15:
                        if rps_result[0] == 'five':
                            return 0
    
    # 아니오1
    if len(landmarks2) > 1: # 손이 2개 있을 때
        if landmarks2[0].landmark[8].y < landmarks2[0].landmark[7].y and landmarks2[1].landmark[8].y < landmarks2[1].landmark[7].y:
            if abs(landmarks2[0].landmark[12].y - landmarks2[0].landmark[16].y) < 0.05 and abs(landmarks2[0].landmark[16].y - landmarks2[0].landmark[20].y) < 0.05:
                if landmarks2[0].landmark[8].x < landmarks2[0].landmark[4].x and landmarks2[1].landmark[4].x < landmarks2[1].landmark[8].x:
                    return 1
    # 아니오2
    if len(landmarks2) > 1:
        if landmarks2[0].landmark[4].x < landmarks2[0].landmark[8].x and landmarks2[1].landmark[8].x < landmarks2[1].landmark[4].x: # 양손 엄지 검지 위치
            if landmarks2[0].landmark[8].y > landmarks2[0].landmark[7].y and landmarks2[1].landmark[8].y > landmarks2[1].landmark[7].y:
                if landmarks2[1].landmark[4].x - landmarks2[0].landmark[4].x > 0.5:
                    return 2

                
        
    # 싫어
    
    if len(landmarks2) > 0:
        if abs(landmarks2[0].landmark[4].y - landmarks2[0].landmark[8].y) < 0.03: # 엄지 검지 거리 
            if abs(landmarks2[0].landmark[4].y - landmarks[10].y) < 0.15:
                if abs(landmarks[10].x - landmarks2[0].landmark[8].x) < 0.15 and abs(landmarks[9].x - landmarks2[0].landmark[4].x) < 0.15:
                    if landmarks2[0].landmark[8].y <landmarks2[0].landmark[6].y:
                        return 3
    
    # 괜찮습니다
    
    if len(landmarks2) > 0:
        if landmarks2[0].landmark[20].y < landmarks2[0].landmark[16].y and landmarks2[0].landmark[20].y < landmarks2[0].landmark[8].y:
            if landmarks[10].x < landmarks2[0].landmark[20].x and landmarks2[0].landmark[20].x < landmarks[9].x and landmarks[10].y < landmarks2[0].landmark[20].y:
                if abs(landmarks2[0].landmark[6].y - landmarks2[0].landmark[10].y) < 0.05 and abs(landmarks2[0].landmark[10].y - landmarks2[0].landmark[14].y) < 0.05:
                    return 4
                
    # 감사합니다
    if len(landmarks2) > 1:
        if abs(landmarks2[1].landmark[5].y - landmarks2[1].landmark[9].y) < 0.1 and abs(landmarks2[1].landmark[13].y - landmarks2[1].landmark[17].y) < 0.1: # 아래 손 수평
            if abs(landmarks2[0].landmark[8].x - landmarks2[0].landmark[12].x) < 0.05 and abs(landmarks2[0].landmark[16].x - landmarks2[0].landmark[12].x) < 0.05: # 윗손 수직
                
                if landmarks2[0].landmark[4].y < landmarks2[0].landmark[8].y:
                    if abs(landmarks2[1].landmark[5].x - landmarks2[0].landmark[20].x) < 0.1:
                        return 5
    
    # 미안합니다
    if len(landmarks2) > 0:
        if rps_result[0] == 'ok':
            if landmarks[4].y > landmarks2[0].landmark[4].y:
                return 6
    
    # 좋다
    
    if len(landmarks2) == 1:
        if abs(landmarks[0].x - landmarks2[0].landmark[5].x) < 0.03 and abs(landmarks[0].y - landmarks2[0].landmark[5].y) < 0.03:
            if abs(landmarks2[0].landmark[6].x - landmarks2[0].landmark[10].x) < 0.03 and   abs(landmarks2[0].landmark[14].x - landmarks2[0].landmark[18].x) < 0.03:
                return 7