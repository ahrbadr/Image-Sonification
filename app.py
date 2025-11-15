import os
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Optional

from flask import Flask, render_template, request, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename

# For image processing
from PIL import Image
import cv2

# For audio writing
from scipy.io.wavfile import write as wav_write

# For visualizations
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_parameters(duration: float, base_freq: float, max_size: int, reverb_strength: float) -> Tuple[bool, str]:
    """Validate input parameters."""
    if duration <= 0 or duration > 60:
        return False, "Duration must be between 0.1 and 60 seconds"
    if base_freq < 20 or base_freq > 20000:
        return False, "Base frequency must be between 20 and 20000 Hz"
    if max_size < 16 or max_size > 2048:
        return False, "Max size must be between 16 and 2048 pixels"
    if reverb_strength < 0 or reverb_strength > 1:
        return False, "Reverb strength must be between 0 and 1"
    return True, ""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Validate upload
        if 'image' not in request.files:
            flash('No file part in request', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                # Get and validate parameters
                duration = float(request.form.get('duration', 5))
                base_freq = float(request.form.get('base_freq', 220))
                waveform = request.form.get('waveform', 'sine')
                scan_type = request.form.get('scan_type', 'vertical')
                max_size = int(request.form.get('max_size', 512))
                reverb_strength = float(request.form.get('reverb_strength', 0.0))

                # Validate parameters
                is_valid, error_msg = validate_parameters(duration, base_freq, max_size, reverb_strength)
                if not is_valid:
                    flash(error_msg, 'error')
                    return redirect(request.url)

                # Generate sonification
                audio_file, scan_path_img, waveform_img = generate_sonification(
                    image_path=upload_path,
                    duration=duration,
                    base_freq=base_freq,
                    waveform=waveform,
                    scan_type=scan_type,
                    max_size=max_size,
                    reverb_strength=reverb_strength
                )

                return render_template('result.html',
                                    audio_file=audio_file,
                                    scan_path_img=scan_path_img,
                                    waveform_img=waveform_img)
            
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, BMP, TIFF).', 'error')
            return redirect(request.url)

    return render_template('index.html')

def generate_piano_note(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a piano-like note with ADSR envelope."""
    t_note = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # ADSR envelope
    attack_time = 0.01
    decay_time = 0.1
    sustain_level = 0.7
    release_time = 0.2
    
    envelope = np.zeros_like(t_note)
    
    # Attack
    attack_samples = int(attack_time * sample_rate)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    decay_samples = int(decay_time * sample_rate)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
    
    # Sustain
    sustain_start = attack_samples + decay_samples
    envelope[sustain_start:] = sustain_level
    
    # Release
    release_samples = int(release_time * sample_rate)
    if release_samples > 0 and release_samples < len(envelope):
        release_start = len(envelope) - release_samples
        envelope[release_start:] = np.linspace(sustain_level, 0, release_samples)
    
    # Generate harmonic-rich tone
    harmonics = [
        (1.0, 1),    # Fundamental
        (0.5, 2),    # 2nd harmonic
        (0.3, 3),    # 3rd harmonic
        (0.2, 4),    # 4th harmonic
        (0.1, 5)     # 5th harmonic
    ]
    
    note_wave = np.zeros_like(t_note)
    for amplitude, harmonic in harmonics:
        note_wave += amplitude * np.sin(2 * np.pi * freq * harmonic * t_note)
    
    return envelope * note_wave

def generate_sonification(image_path: str, duration: float, base_freq: float, 
                         waveform: str, scan_type: str, max_size: int, 
                         reverb_strength: float) -> Tuple[str, str, str]:
    """
    Process the image and generate the audio and visualization.
    """
    try:
        # --- Load and Resize Image ---
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        h, w = img_np.shape

        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img_np.shape

        sample_rate = 44100
        total_samples = int(sample_rate * duration)

        # --- Generate Audio ---
        if waveform == 'piano':
            # Discrete note synthesis
            note_duration = 0.3  # seconds per note
            note_samples = int(sample_rate * note_duration)
            num_notes = int(duration / note_duration)
            audio_float = np.zeros(total_samples, dtype=np.float32)

            # Determine scan direction
            if scan_type == 'horizontal':
                positions = np.linspace(0, h - 1, num_notes)
                is_horizontal = True
            else:
                positions = np.linspace(0, w - 1, num_notes)
                is_horizontal = False

            num_piano_keys = 88  # keys from A0 (27.5Hz) to C8 (~4186Hz)
            
            for i in range(num_notes):
                pos = int(positions[i])
                if is_horizontal:
                    brightness = img_np[pos, :].mean() / 255.0
                else:
                    brightness = img_np[:, pos].mean() / 255.0

                # Map brightness to a piano key (discrete)
                key_index = int(round(brightness * (num_piano_keys - 1)))
                freq = 27.5 * (2 ** (key_index / 12))

                # Generate piano note
                note_wave = generate_piano_note(freq, note_duration, sample_rate)
                
                start = i * note_samples
                end = start + len(note_wave)
                if end > total_samples:
                    end = total_samples
                    note_wave = note_wave[:end - start]
                
                audio_float[start:end] += note_wave

        else:
            # Continuous waveform generation
            t = np.linspace(0, duration, total_samples, endpoint=False)
            audio_float = np.zeros_like(t, dtype=np.float32)

            if scan_type == 'horizontal':
                path = np.linspace(0, h - 1, total_samples)
                is_horizontal = True
            else:
                path = np.linspace(0, w - 1, total_samples)
                is_horizontal = False

            for i in range(total_samples):
                pos = int(path[i])
                if is_horizontal:
                    brightness = img_np[pos, :].mean() / 255.0
                else:
                    brightness = img_np[:, pos].mean() / 255.0

                freq = base_freq * (1.0 + 2.0 * brightness)  # Expanded frequency range
                phase = 2.0 * np.pi * freq * t[i]
                
                if waveform == 'square':
                    val = 1.0 if np.sin(phase) >= 0 else -1.0
                elif waveform == 'sawtooth':
                    frac = (phase % (2.0 * np.pi)) / (2.0 * np.pi)
                    val = 2.0 * frac - 1.0
                elif waveform == 'triangle':
                    frac = (phase % (2.0 * np.pi)) / (2.0 * np.pi)
                    val = 2.0 * abs(2.0 * frac - 1.0) - 1.0
                else:  # sine
                    val = np.sin(phase)
                
                amplitude = 0.3 + 0.7 * brightness  # Dynamic amplitude based on brightness
                audio_float[i] = amplitude * val

        # --- Optional Reverb ---
        if reverb_strength > 0:
            impulse_len = int(sample_rate * 0.3)
            impulse = np.random.randn(impulse_len).astype(np.float32) * 0.1
            impulse = impulse * np.exp(-np.linspace(0, 5, impulse_len))  # Exponential decay
            conv = np.convolve(audio_float, impulse, mode='full')[:len(audio_float)]
            audio_float = conv * reverb_strength + audio_float * (1.0 - reverb_strength)

        # --- Normalize and Convert ---
        max_val = np.max(np.abs(audio_float))
        if max_val < 1e-9:
            max_val = 1.0
        audio_float /= max_val
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # --- Save Audio ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"sonified_{timestamp}.wav"
        audio_path = os.path.join(app.config['STATIC_FOLDER'], audio_filename)
        wav_write(audio_path, sample_rate, audio_int16)

        # --- Create Visualizations ---
        scan_path_img, waveform_img = create_visualizations(img_np, scan_type, audio_float, timestamp, 
                                                          waveform == 'piano', len(positions) if waveform == 'piano' else total_samples)

        return audio_filename, scan_path_img, waveform_img

    except Exception as e:
        logger.error(f"Error in generate_sonification: {str(e)}")
        raise

def create_visualizations(img_np: np.ndarray, scan_type: str, audio_float: np.ndarray, 
                         timestamp: str, is_piano: bool, num_points: int) -> Tuple[str, str]:
    """Create scan path and waveform visualizations."""
    h, w = img_np.shape
    
    # Scan path overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np, cmap='gray')
    
    if scan_type == 'horizontal':
        if is_piano:
            y_positions = np.linspace(0, h - 1, num_points)
            plt.plot(np.linspace(0, w - 1, num_points), y_positions, 'r-', alpha=0.7, linewidth=2)
            plt.scatter(np.linspace(0, w - 1, num_points), y_positions, c='red', s=20, alpha=0.8)
        else:
            y_path = np.linspace(0, h - 1, num_points)
            plt.plot(np.linspace(0, w - 1, num_points), y_path, 'r-', alpha=0.5, linewidth=1)
    else:
        if is_piano:
            x_positions = np.linspace(0, w - 1, num_points)
            plt.plot(x_positions, np.linspace(0, h - 1, num_points), 'r-', alpha=0.7, linewidth=2)
            plt.scatter(x_positions, np.linspace(0, h - 1, num_points), c='red', s=20, alpha=0.8)
        else:
            x_path = np.linspace(0, w - 1, num_points)
            plt.plot(x_path, np.linspace(0, h - 1, num_points), 'r-', alpha=0.5, linewidth=1)
    
    plt.title("Image Scan Path", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    scan_path_img = f"scanpath_{timestamp}.png"
    sp_save_path = os.path.join(app.config['STATIC_FOLDER'], scan_path_img)
    plt.savefig(sp_save_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()

    # Waveform preview
    plt.figure(figsize=(10, 4))
    seg_len = min(2000, len(audio_float))
    time_axis = np.linspace(0, seg_len / 44100, seg_len)
    plt.plot(time_axis, audio_float[:seg_len], 'b-', linewidth=1)
    plt.title("Audio Waveform Preview", fontsize=14, pad=20)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    waveform_img = f"waveform_{timestamp}.png"
    wf_save_path = os.path.join(app.config['STATIC_FOLDER'], waveform_img)
    plt.savefig(wf_save_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()

    return scan_path_img, waveform_img

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    return "File too large. Please upload an image smaller than 16MB.", 413

@app.errorhandler(500)
def internal_error(e):
    return "Internal server error. Please try again.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)