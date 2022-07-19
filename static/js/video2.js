let video = document.getElementById('video');
let canvas = document.getElementById('canvas');


var requestId = undefined;
// canvas 좌우 반전을 위해 video -> canvas로 그리기
function loop(){ 
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    requestId = requestAnimationFrame(loop); 
};

function data_ajax(){
    var dataURL = canvas.toDataURL();
    // 화면을 갱신하지 않고 정보 전송
    $.ajax({
        type: "POST",
        url: "canvas_image",
        data:{
        imageBase64: dataURL
        }
    }).done(function(text) {
        console.log(text);
        console.log(text, keys[number])
        // 맞으면 다음으로 진행
        // 끝까지 다하면 종료
        if (text == keys[number]){
            
            number = number + 1;
            if (number == keys.length){
                start.style.display = 'block';
                source.src = "";
                edu_video.load();
                
                document.getElementById('sign_text').innerHTML="";
                document.getElementById('n__1__________________________').innerHTML="";
            }
            else{
                now_text();
            }
            
        }
    });
}
