const selectImageBtn = document.getElementById('selectImageBtn');
const fileInput = document.getElementById('fileInput');
const recognizeBtn = document.getElementById('recognizeBtn');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const saveBtn = document.getElementById('saveBtn');
const imagePreview = document.getElementById('imagePreview');
const recognizedText = document.getElementById('recognizedText');
const statusText = document.getElementById('statusText');

let selectedImage = null;

selectImageBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', handleImageSelect);
recognizeBtn.addEventListener('click', handleRecognizeText);
clearBtn.addEventListener('click', handleClear);
copyBtn.addEventListener('click', handleCopy);
saveBtn.addEventListener('click', handleSave);

function handleImageSelect(event) {
    const file = event.target.files[0];
    
    if (file && file.type.startsWith('image/')) {
        selectedImage = file;
        displayImage(file);
        updateStatus('Image loaded successfully');
    } else {
        updateStatus('Please select a valid image file');
    }
}

function displayImage(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Selected image">`;
    };
    
    reader.readAsDataURL(file);
}

function handleRecognizeText() {
    if (!selectedImage) {
        updateStatus('Please select an image first');
        return;
    }
    
    updateStatus('Recognizing text...');
    recognizeBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', selectedImage);
    
    fetch('/recognize', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            updateStatus('Error: ' + data.error);
            recognizedText.value = 'Error occurred during text recognition.';
        } else {
            recognizedText.value = data.text;
            updateStatus('Text recognition complete');
        }
        recognizeBtn.disabled = false;
    })
    .catch(error => {
        updateStatus('Error: ' + error.message);
        recognizedText.value = 'Failed to recognize text.';
        recognizeBtn.disabled = false;
    });
}

function handleClear() {
    selectedImage = null;
    fileInput.value = '';
    imagePreview.innerHTML = `
        <p class="placeholder-text">沒有選擇任何圖片</p>
        <p class="placeholder-subtext">點擊"上傳圖片"以開始</p>
    `;
    recognizedText.value = '';
    updateStatus('已清除');
}

function handleCopy() {
    if (recognizedText.value) {
        recognizedText.select();
        document.execCommand('copy');
        updateStatus('文字已複製到剪貼簿');
    } else {
        updateStatus('沒有可複製的文字');
    }
}

function handleSave() {
    if (recognizedText.value) {
        const blob = new Blob([recognizedText.value], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'recognized-text.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        updateStatus('Text saved to file');
    } else {
        updateStatus('No text to save');
    }
}

function updateStatus(message) {
    statusText.textContent = message;
    setTimeout(() => {
        statusText.textContent = 'Ready';
    }, 3000);
}

imagePreview.addEventListener('dragover', (e) => {
    e.preventDefault();
    imagePreview.style.borderColor = '#0e639c';
});

imagePreview.addEventListener('dragleave', (e) => {
    e.preventDefault();
    imagePreview.style.borderColor = '#3c3c3c';
});

imagePreview.addEventListener('drop', (e) => {
    e.preventDefault();
    imagePreview.style.borderColor = '#3c3c3c';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedImage = file;
        displayImage(file);
        updateStatus('Image loaded successfully');
    }
});