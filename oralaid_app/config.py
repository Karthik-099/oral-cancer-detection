"""
OralAID Configuration File
Centralized configuration for the application
"""

import os
from pathlib import Path

class Config:
    """Base configuration"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'oralaid-secret-key-2024'
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///oralaid.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'models'
    CLASSIFIER_MODEL = 'final_efficientnet_b4_oral_cancer.pth'
    SEGMENTER_MODEL = 'unet_segmenter.pth'
    
    # AI settings
    IMAGE_SIZE = (224, 224)
    SEGMENTATION_SIZE = (512, 512)
    CONFIDENCE_THRESHOLD = 0.5
    
    # Results storage
    RESULTS_FOLDER = 'static/results'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = 'logs/oralaid.log'
    
    # Performance settings
    GRADCAM_ENABLED = True
    LIME_ENABLED = True
    SEGMENTATION_ENABLED = True
    LIME_NUM_SAMPLES = 1000
    
    # Clinical settings
    RISK_MAPPING = {
        'normal': {'category': 'Non-suspicious', 'risk': 'Low-risk', 'color': '#28a745'},
        'keratosis': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
        'leukoplakia_homogenous': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
        'leukoplakia_speckled': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
        'erythroplakia': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
        'proliferative_verrucous': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'},
        'lichen_planus': {'category': 'Suspicious', 'risk': 'Low-risk', 'color': '#ffc107'},
        'squamous_cell_carcinoma': {'category': 'Suspicious', 'risk': 'High-risk', 'color': '#dc3545'}
    }
    
    # API settings
    API_RATE_LIMIT = "100 per hour"
    API_TIMEOUT = 30
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        directories = [
            Config.UPLOAD_FOLDER,
            Config.RESULTS_FOLDER,
            Config.MODEL_PATH,
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Database connection pooling
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}