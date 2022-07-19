let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

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
        url: "canvas_image2",
        data:{
        imageBase64: dataURL
        }
    }).done(function(text) {
        console.log(text);
    });
}

// 버튼 클릭시 
canvas.addEventListener('click', ()=>{
    //웹캠이 꺼져있을 때
    if (video.paused) {
        // 웹캠 시작
        navigator.mediaDevices.getUserMedia({ video: true })
        .then( (stream) => {
            video.srcObject  = stream  //비디오 테그에 웹캠 스트림을 넣습니다.
            video.play()  //비디오 테그 실행

            //그리기 반복
            requestId = window.requestAnimationFrame(loop); 
            interval = setInterval((data_ajax),500)
            
        })
        .catch( (error)=>{
            console.log(error);
        });

    //웹캠이 켜져있을 때
    } else { 

        video.srcObject.getTracks().forEach(function(track) {
            track.stop();
        });
        
        video.pause(); 
        clearInterval(interval);

        ctx.strokeStyle = "red"; // border(선)색
        ctx.lineWidth = "20";
        ctx.fillStyle = "#c1c1c1"; // 면색상
        
        //드로잉    
        window.cancelAnimationFrame(requestId);    
        requestId = undefined;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // edu_video_btn.style.display = 'block';

    } 
    
})