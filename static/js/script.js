// File upload handling
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('image');
    const fileInfo = document.getElementById('file-info');
    const reverbSlider = document.getElementById('reverb_strength');
    const reverbValue = document.getElementById('reverb_value');

    // File input change handler
    if (fileInput && fileInfo) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                fileInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
                
                // Validate file size (16MB limit)
                if (file.size > 16 * 1024 * 1024) {
                    alert('File size must be less than 16MB');
                    fileInput.value = '';
                    fileInfo.textContent = 'No file selected';
                }
            } else {
                fileInfo.textContent = 'No file selected';
            }
        });
    }

    // Reverb slider value display
    if (reverbSlider && reverbValue) {
        reverbSlider.addEventListener('input', function() {
            reverbValue.textContent = parseFloat(this.value).toFixed(1);
        });
    }

    // Form validation
    const form = document.querySelector('.upload-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const duration = document.getElementById('duration').value;
            const baseFreq = document.getElementById('base_freq').value;
            
            if (duration <= 0 || duration > 60) {
                e.preventDefault();
                alert('Duration must be between 0.1 and 60 seconds');
                return;
            }
            
            if (baseFreq < 20 || baseFreq > 20000) {
                e.preventDefault();
                alert('Base frequency must be between 20 and 20000 Hz');
                return;
            }
        });
    }

    // Add loading state to submit button
    const submitBtn = document.querySelector('.submit-btn');
    if (submitBtn) {
        submitBtn.addEventListener('click', function() {
            const file = fileInput.files[0];
            if (file) {
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                this.disabled = true;
            }
        });
    }
});

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Audio player enhancements
function enhanceAudioPlayer() {
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach(audio => {
        // Add custom controls if needed
        audio.addEventListener('loadedmetadata', function() {
            console.log(`Audio duration: ${this.duration.toFixed(2)} seconds`);
        });
    });
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    enhanceAudioPlayer();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});