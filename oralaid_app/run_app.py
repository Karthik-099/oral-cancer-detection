#!/usr/bin/env python3
"""
OralAID Application Runner
Simple script to start the OralAID web application
"""

import os
import sys

print("=" * 60)
print(" OralAID - AI-Powered Oral Cancer Detection")
print("=" * 60)
print()

# Change to app directory
os.chdir('/home/karthik/oral-cancer-detection/oralaid_app')

# Import and run the app
try:
    from app import app
    
    print("✅ All dependencies loaded successfully")
    print()
    print("🌐 Starting web server...")
    print("📍 Access the application at: http://localhost:5000")
    print("👤 Login credentials: admin / admin123")
    print()
    print("📋 Features Available:")
    print("   ✓ AI-powered oral cancer detection")
    print("   ✓ LIME & GradCAM visualizations")
    print("   ✓ Risk assessment & recommendations")
    print("   ✓ Patient management")
    print("   ✓ Clinical reporting")
    print()
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
    
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure all dependencies are installed:")
    print("   pip install flask flask-sqlalchemy flask-login torch torchvision pillow numpy opencv-python")
    print("2. Check if port 5000 is available")
    print("3. Ensure you're in the correct directory")
    sys.exit(1)
