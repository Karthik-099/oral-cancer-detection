#!/usr/bin/env python3
"""
OralAID Application Launcher
Run this script to start the OralAID web application
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'torch', 'torchvision', 'PIL', 'numpy', 
        'opencv-python', 'matplotlib', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'static/results',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created necessary directories")

def check_models():
    """Check if AI models are available"""
    model_files = [
        'models/final_efficientnet_b4_oral_cancer.pth',
        'models/unet_segmenter.pth'
    ]
    
    missing_models = []
    for model_file in model_files:
        if not Path(model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️  AI model files not found:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n📥 The application will use fallback models")
        print("   For full functionality, please add the trained models to the models/ directory")
    else:
        print("✅ AI models found")

def start_application():
    """Start the Flask application"""
    print("🚀 Starting OralAID Web Application...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Start Flask app
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        print("🌐 Opening web browser...")
        webbrowser.open('http://localhost:5000')
        
        print("\n✅ OralAID is running!")
        print("📱 Web Interface: http://localhost:5000")
        print("👤 Demo Login: admin / admin123")
        print("\n⏹️  Press Ctrl+C to stop the application")
        
        # Wait for process to complete
        for line in process.stdout:
            print(line.strip())
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping OralAID...")
        process.terminate()
        print("✅ Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function"""
    print("🦷 OralAID - AI-Powered Oral Cancer Detection")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check models
    check_models()
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()