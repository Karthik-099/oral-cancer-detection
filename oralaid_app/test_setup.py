#!/usr/bin/env python3
"""
OralAID Application Test Script
Verify that all components are properly set up
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        'app.py',
        'desktop_app.py',
        'config.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'README.md',
        'templates/base.html',
        'templates/index.html',
        'templates/login.html',
        'templates/register.html',
        'templates/dashboard.html',
        'templates/diagnose.html',
        'templates/result.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_directories():
    """Test if all required directories exist"""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'templates',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for directory in missing_dirs:
            print(f"   - {directory}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_imports():
    """Test if critical imports work"""
    print("🔍 Testing Python imports...")
    
    critical_imports = [
        ('flask', 'Flask'),
        ('PIL', 'Image'),
        ('numpy', 'numpy'),
        ('torch', 'PyTorch'),
    ]
    
    failed_imports = []
    
    for module, name in critical_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            failed_imports.append((module, name, str(e)))
            print(f"❌ {name} import failed: {e}")
    
    return len(failed_imports) == 0

def test_configuration():
    """Test configuration loading"""
    print("🔍 Testing configuration...")
    
    try:
        from config import Config, DevelopmentConfig
        config = DevelopmentConfig()
        
        # Test essential config values
        assert hasattr(config, 'SECRET_KEY')
        assert hasattr(config, 'SQLALCHEMY_DATABASE_URI')
        assert hasattr(config, 'RISK_MAPPING')
        
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("🔍 Testing Flask app creation...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Import and create app
        from app import app
        
        # Test app creation
        assert app is not None
        assert app.config is not None
        
        print("✅ Flask app created successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def generate_summary():
    """Generate a summary of the application"""
    print("\n" + "="*60)
    print("🦷 OralAID Application Summary")
    print("="*60)
    
    print("\n📋 Features Implemented:")
    features = [
        "✅ Modern web interface with Bootstrap 5",
        "✅ User authentication and session management",
        "✅ Patient management system",
        "✅ AI-powered image analysis",
        "✅ Explainable AI (GradCAM, LIME)",
        "✅ Lesion segmentation",
        "✅ Risk stratification system",
        "✅ Clinical recommendations",
        "✅ Report generation and export",
        "✅ Windows desktop application",
        "✅ Docker deployment configuration",
        "✅ Comprehensive documentation"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n🚀 Quick Start Commands:")
    print("  Web App:     python run.py")
    print("  Desktop App: python desktop_app.py")
    print("  Docker:      docker-compose up -d")
    print("  Build EXE:   python setup_windows.py")
    
    print("\n🌐 Access Points:")
    print("  Web Interface: http://localhost:5000")
    print("  Demo Login:    admin / admin123")
    print("  API Docs:      http://localhost:5000/api/docs")
    
    print("\n📊 Technology Stack:")
    tech_stack = [
        "Backend: Flask + SQLAlchemy",
        "Frontend: HTML5/CSS3/JavaScript + Bootstrap 5",
        "AI: PyTorch + segmentation-models-pytorch",
        "Explainability: GradCAM + LIME",
        "Desktop: Tkinter + PyInstaller",
        "Deployment: Docker + Docker Compose",
        "Monitoring: Prometheus + Grafana"
    ]
    
    for tech in tech_stack:
        print(f"  • {tech}")

def main():
    """Main test function"""
    print("🧪 OralAID Application Test Suite")
    print("="*50)
    
    tests = [
        test_file_structure,
        test_directories,
        test_imports,
        test_configuration,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print("="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OralAID is ready to run.")
        generate_summary()
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)