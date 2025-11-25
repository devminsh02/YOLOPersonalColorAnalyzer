const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const captureButton = document.getElementById('captureButton');

// 웹캠 시작 함수
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('웹캠 시작 중 에러 발생:', error);
        alert('웹캠에 접근할 수 없습니다.');
    }
}

// 웹캠 시작/중지 버튼 핸들러
startButton.addEventListener('click', () => {
    if (startButton.textContent === 'Start') {
        startWebcam();
        startButton.textContent = 'Stop';
        captureButton.disabled = false;
    } else {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        startButton.textContent = 'Start';
        captureButton.disabled = true;
    }
});

// 이미지 캡처 및 다운로드/서버 전송
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 캡처 이미지를 서버로 전송
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'captured_image.png');

        fetch('/diagnosis', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text();
            })
            .then(result => {
                console.log('서버 응답:', result);
                alert('진단이 완료되었습니다!');
                // 결과 페이지로 이동 또는 다른 작업 수행
            })
            .catch(error => {
                console.error('이미지 전송 중 에러 발생:', error);
                alert('이미지를 전송하는 중 문제가 발생했습니다.');
            });
    });
});
