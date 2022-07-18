


// 태그 변수에 할당
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
console.log("되나?")
// 웹캠 시작
navigator.mediaDevices.getUserMedia({ video: true })
.then( (stream) => {
    video.srcObject  = stream  //비디오 테그에 웹캠 스트림을 넣습니다.
    video.play()  //비디오 테그 실행

    video.addEventListener('play', ()=>{
        // 일정 시간마다 ajax 실행
        setInterval(()=>{

            // 데이터 전송을 위해 형태 변환
            var dataURL = canvas.toDataURL();

            // 화면을 갱신하지 않고 정보 전송
            $.ajax({
                type: "POST",
                url: "canvas_image",
                data:{
                imageBase64: dataURL
                }
            }).done(function() {
                //alert('sent');
            });

        },500)

    }, false);
})
.catch( (error)=>{
      console.log(error);
});


// canvas 좌우 반전을 위해 video -> canvas로 그리기
(function loop(){
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    

    

    requestAnimationFrame(loop); //그리기 반복

})();
console.log("되나?2")