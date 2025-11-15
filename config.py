import os

# Basic Configuration
SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Application Settings
DEBUG = False
TESTING = False

# Audio Settings
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION = 5
DEFAULT_BASE_FREQ = 220
DEFAULT_MAX_SIZE = 512