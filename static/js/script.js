// DOM Elements
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

// Event Listeners
selectImageBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', handleImageSelect);
recognizeBtn.addEventListener('click', handleRecognizeText);
clearBtn.addEventListener('click', handleClear);
copyBtn.addEventListener('click', handleCopy);
saveBtn.addEventListener('click', handleSave);

// Handle Image Selection
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

// Display Selected Image
function displayImage(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Selected image">`;
    };
    
    reader.readAsDataURL(file);
}

// Handle Text Recognition
function handleRecognizeText() {
    if (!selectedImage) {
        updateStatus('Please select an image first');
        return;
    }
    
    updateStatus('Recognizing text...');
    recognizeBtn.disabled = true;
    
    // Create FormData to send the image to the backend
    const formData = new FormData();
    formData.append('image', selectedImage);
    
    // Send POST request to Flask backend
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

// Handle Clear
function handleClear() {
    selectedImage = null;
    fileInput.value = '';
    imagePreview.innerHTML = `
        <p class="placeholder-text">No image selected</p>
        <p class="placeholder-subtext">Click 'Select Image' to get started</p>
    `;
    recognizedText.value = '';
    updateStatus('Cleared');
}

// Handle Copy
function handleCopy() {
    if (recognizedText.value) {
        recognizedText.select();
        document.execCommand('copy');
        updateStatus('Text copied to clipboard');
        
        // Alternative modern approach (may require HTTPS)
        // navigator.clipboard.writeText(recognizedText.value).then(() => {
        //     updateStatus('Text copied to clipboard');
        // });
    } else {
        updateStatus('No text to copy');
    }
}

// Handle Save
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

// Update Status Bar
function updateStatus(message) {
    statusText.textContent = message;
    setTimeout(() => {
        statusText.textContent = 'Ready';
    }, 3000);
}

// Optional: Add drag and drop functionality
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