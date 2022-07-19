var direction = document.getElementById('direction');
var answer = document.getElementById('answer');

answer.innerHTML = "";
var preWord;

function data_ajax_input(){
    $.ajax({
        type: "POST",
        url: "text_input",
        data:{
        
        }
    }).done(function(text) {
        preWord = answer.innerHTML.split(' ');
        preWord = preWord[preWord.length -1];
        console.log(preWord)
        if (preWord != text) {
            answer.innerHTML += ' ' + text
        }
        
    })
}

setInterval((data_ajax_input),3000)