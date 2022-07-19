var start = document.getElementById('start');

var source = document.getElementById('source');
var edu_video = document.getElementById('n_20');

const array = {
    '너는' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191014/627265/MOV000251996_700X466.webm',
    '이름' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191015/627715/MOV000256668_700X466.webm',
    '무엇' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191028/632014/MOV000257351_700X466.webm',
    '어디' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/732429/MOV000243844_700X466.webm',
    '살다' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191021/629642/MOV000254514_700X466.webm',
    '좋다' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200824/735157/MOV000242450_700X466.webm',
    '음식,먹다' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191025/630761/MOV000250824_700X466.webm',
    '어제' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191021/629532/MOV000254995_700X466.webm',
    '놀다' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191011/626620/MOV000241795_700X466.webm',
    '나' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191028/631984/MOV000248548_700X466.webm',
    '같이' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191001/623709/MOV000236380_700X466.webm',
    '가자' : 'http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191028/632050/MOV000249486_700X466.webm'

}
const array2 = {
    '너는' : '오른 주먹의 1지를 펴서 끝이 밖으로 향하게 하여 약간 내민다.',
    '이름' : '오른 주먹의 1·5지를 펴서 끝이 왼쪽으로 향하게 하여 왼쪽 가슴에 1·5지 옆면이 닿게 댄다.',
    '무엇' : '오른 주먹의 1지를 펴서 바닥이 밖으로 향하게 세워 좌우로 두 번 흔든다.',
    '어디' : '오른 주먹의 1지를 펴서 바닥이 밖으로 향하게 세워 왼쪽으로 흔든 다음, 손바닥이 아래로 향하게 반쯤 구부린 오른손을 가슴 앞에서 약간 내리다가 멈춘다.',
    '살다' : '1·5지를 펴서 바닥이 밖으로 향하게 쥔 두 주먹을 동시에 오른쪽으로 한 바퀴 돌린다.',
    '좋다' : '오른 주먹을 코에 1·5지 옆면이 닿게 대고 팔을 좌우로 두 번 흔든다.',
    '음식,먹다' : '오른손을 펴서, 손바닥이 위로 향하게 하여 두 번 입으로 올린다.',
    '어제' : '오른 주먹의 1지를 펴서 세워 오른쪽 어깨 너머로 넘긴다.',
    '놀다' : '손등이 위로 향하게 편다. 오른손과 손바닥이 위로 향하게 편다. 왼손이 가까이 마주 보게 하여 서로 엇갈리게 두 바퀴 돌린다.',
    '나' : '오른 손바닥을 가슴 중앙에 댄다.',
    '같이' : '두 주먹의 1지를 펴서 끝이 밖으로 바닥이 위를 향하게 하였다가 바닥이 아래로 향하게 중앙으로 반원을 그리며 돌려 맞댄다.',
    '가자' : '오른손을 펴서 손등이 밖으로 손끝이 아래로 향하게 하여 내밀며 손끝을 약간 들어 올린다'

}







function shuffle(array) {
    array.sort(() => Math.random() - 0.5);
}






var keys = Object.keys(array)

shuffle(keys)

var number = 0;

function now_text(){
    source.src = array[keys[number]]
    edu_video.load();
    edu_video.play();
    
    document.getElementById('sign_text').innerHTML=keys[number];
    document.getElementById('n__1__________________________').innerHTML=array2[keys[number]];

    document.getElementById('n_1_bq').src= '/static/images/' + keys[number] + '.jpg'

}





// 시작 눌렀을 때 10개 단어 랜덤 순서 
start.addEventListener('click', ()=>{

    // 버튼 사라짐
    start.style.display = 'none';

    //선택 단어 영상 출력

    
    now_text();

    

    

    // let rPick = Math.floor(Math.random() * keys.length);
    
    // source.src = array[keys[rPick]]
    // edu_video.load();
    // edu_video.play();
    
    
    // document.getElementById('sign_text').innerHTML=keys[rPick];
    // document.getElementById('n__1__________________________').innerHTML=array2[keys[rPick]];



})
