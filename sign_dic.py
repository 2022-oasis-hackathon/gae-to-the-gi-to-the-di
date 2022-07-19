
# pose : landmark 
# hand : landmark2


array = {
    0 : '네',
    1 : '아니오1',
    2 : '싫어',
    3 : '괜찮아',
    4 : '감사합니다',
    5 : '미안합니다',
    6 : '좋다',
    7 : '너',
    8 : '이름',
    9 : '무엇',
    10 : '어디',
    11 : '살다',
    12 : '먹다',
    13 : '어제',
    14 : '놀다',
    15 : '나',
    16 : '같이',
    17 : '가자'
    
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
                    if landmarks2[0].landmark[8].y > landmarks[0].y and landmarks2[1].landmark[8].y > landmarks[0].y:
                        
                        if abs(landmarks2[0].landmark[8].x - landmarks2[1].landmark[8].x) > 0.1:
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
                        if abs(landmarks2[0].landmark[4].y - landmarks[0].y) > 0.1:
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
                    if abs(landmarks2[1].landmark[5].x - landmarks2[0].landmark[20].x) < 0.05:
                    
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
            
            
            
    # 너
    if len(landmarks2) == 1:
        if rps_result[0] != 'five':
            if landmarks2[0].landmark[3].x < landmarks2[0].landmark[8].x:
                if abs(landmarks2[0].landmark[8].x - landmarks2[0].landmark[7].x) < 0.02 and abs(landmarks2[0].landmark[6].x - landmarks2[0].landmark[5].x) < 0.02:
                    if abs(landmarks2[0].landmark[8].y - landmarks2[0].landmark[7].y) < 0.02 and abs(landmarks2[0].landmark[6].y - landmarks2[0].landmark[5].y) < 0.02:
                        return 8
    
    # 이름
    if len(landmarks2) == 1:
        if landmarks2[0].landmark[8].y < landmarks2[0].landmark[10].y and landmarks2[0].landmark[10].y < landmarks2[0].landmark[4].y:
            if abs(landmarks2[0].landmark[8].x - landmarks2[0].landmark[4].x) < 0.1: # 엄지 검지 끝 x축이 비슷
                if landmarks[12].x < landmarks2[0].landmark[8].x and landmarks2[0].landmark[8].x < landmarks[10].x:
                    if landmarks[12].y < landmarks2[0].landmark[8].y:
                        return 9
                
    # 무엇
    if len(landmarks2) == 1:
        if rps_result[0] == 'one':
            if landmarks2[0].landmark[8].y < landmarks2[0].landmark[5].y:
                
                return 10
        
    # 어디
    if len(landmarks2) == 1:
        if landmarks2[0].landmark[4].x < landmarks2[0].landmark[8].x and landmarks2[0].landmark[12].x < landmarks2[0].landmark[16].x and landmarks2[0].landmark[16].x < landmarks2[0].landmark[20].x:
            if landmarks2[0].landmark[4].y > landmarks2[0].landmark[3].y and landmarks2[0].landmark[8].y > landmarks2[0].landmark[7].y and landmarks2[0].landmark[12].y > landmarks2[0].landmark[11].y:
                if landmarks[11].y < landmarks2[0].landmark[0].y:
                    return 11
    # 살다
    if len(landmarks2) > 1:
        if landmarks2[0].landmark[8].y < landmarks2[0].landmark[7].y and landmarks2[1].landmark[8].y < landmarks2[1].landmark[7].y:
            if abs(landmarks2[0].landmark[12].y - landmarks2[0].landmark[16].y) < 0.05 and abs(landmarks2[0].landmark[16].y - landmarks2[0].landmark[20].y) < 0.05:
                if landmarks2[0].landmark[8].x < landmarks2[0].landmark[4].x and landmarks2[1].landmark[4].x < landmarks2[1].landmark[8].x:
                    if landmarks2[0].landmark[8].y < landmarks[0].y and landmarks2[1].landmark[8].y < landmarks[0].y:
                        return 12
    
    # 먹다
    if len(landmarks2) == 1:
        if rps_result[0] == 'five':
            if landmarks[10].x < landmarks2[0].landmark[12].x and landmarks2[0].landmark[12].x < landmarks[9].x:
                if abs(landmarks[10].y - landmarks2[0].landmark[12].y) < 0.02:
                    if landmarks2[0].landmark[12].x < landmarks2[0].landmark[8].x:
                        return 13
    
    
    # 어제
    if len(landmarks2) == 1:
        if rps_result[0] != 'five':
            if landmarks2[0].landmark[8].x >landmarks[7].x:
                if abs(landmarks2[0].landmark[8].y - landmarks[7].y) < 0.2:
                    if landmarks2[0].landmark[8].y < landmarks2[0].landmark[4].y:
                        if abs(landmarks2[0].landmark[8].x - landmarks2[0].landmark[9].x) < 0.15:
                            if abs(landmarks2[0].landmark[8].y - landmarks2[0].landmark[7].y) < 0.01:
                                return 14
    
    # 놀다
    if len(landmarks2) > 1:
        if abs(landmarks[13].y - landmarks[14].y) < 0.07:
            if abs(landmarks2[0].landmark[12].x - landmarks2[1].landmark[0].x) < 0.03 and abs(landmarks2[1].landmark[12].x - landmarks2[0].landmark[0].x) < 0.03:
                return 15
    
    
    
    # 나
    if len(landmarks2) == 1:
        if landmarks2[0].landmark[4].y < landmarks2[0].landmark[8].y:
            if landmarks2[0].landmark[8].x < landmarks[10].x and landmarks2[0].landmark[12].x < landmarks[10].x and landmarks2[0].landmark[16].x < landmarks[10].x:
                if landmarks[9].x < landmarks2[0].landmark[4].x:
                    if abs(landmarks2[0].landmark[4].y - landmarks[0].y) > 0.1:
                        return 16
    
    # 같이
    
    if len(landmarks2) == 2:
        
        if abs(landmarks2[0].landmark[8].x - landmarks2[1].landmark[8].x) < 0.02 and abs(landmarks2[0].landmark[8].y - landmarks2[1].landmark[8].y) < 0.02:
            if landmarks2[0].landmark[8].y < landmarks2[0].landmark[4].y and landmarks2[1].landmark[8].y < landmarks2[1].landmark[4].y:
                return 17
    
    # 가자
    if len(landmarks2) == 1:
        if landmarks2[0].landmark[4].x < landmarks2[0].landmark[8].x:
            if rps_result[0] == 'five':
                if landmarks2[0].landmark[8].x < landmarks2[0].landmark[4].y:
                    return 18
    
    