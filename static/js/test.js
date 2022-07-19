var answer = document.getElementById('answer');
var score = document.getElementById('score');
var word = document.getElementById('word');

answer.innerHTML = "X";
score.innerHTML = "0";
word.innerHTML = "여러분";

// 이미지 부분
// const arr_img = {
//     '여러분' : '',
//     '안녕하세요' : , 
//     '발표' : , 
//     '시작' : , 
//     '어떻게' : , 
//     '오늘' : , 
//     '하루' : , 
//     '보내다' : 
// }

const arr = [ '발표', '여러분', '안녕하세요', '시작', '어떻게', '오늘', '하루', '보내다'];
var idx = 0;

function data_ajax_input(){
    $.ajax({
        type: "POST",
        url: "text_input",
        data:{
        
        }
    }).done(function(text) {
        word.innerHTML = arr[idx];

        if (text == arr[idx]){
            answer.innerHTML = 'O';
            idx += 1;
            score.innerHTML = idx;
        }
        else {
            answer.innerHTML = 'X';
        }
    })
}

    
    
setInterval((data_ajax_input),500)