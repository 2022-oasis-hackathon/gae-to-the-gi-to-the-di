var answer = document.getElementById('answer');


answer.innerHTML = "";

function data_ajax_input(){
    $.ajax({
        type: "POST",
        url: "text_input",
        data:{
        
        }
    }).done(function(text) {
        console.log(text)
        
    })
}

    
    
setInterval((data_ajax_input),500)