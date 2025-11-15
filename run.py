#!/usr/bin/env python3
"""
Run script for the Image Sonification application
"""

from app import app

if __name__ == '__main__':
    print("Starting Image Sonification Server...")
    print("Access the application at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)