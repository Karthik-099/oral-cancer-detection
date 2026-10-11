# OralAID - AI-Powered Oral Cancer Detection System

![OralAID Logo](static/images/oralaid_logo.png)

## 🦷 Overview

OralAID is a comprehensive AI-driven diagnostic system for early detection and classification of oral lesions from intraoral images. Built with cutting-edge deep learning technologies, it provides clinicians with accurate, interpretable AI assistance for oral cancer screening.

### 🎯 Key Features

- **🧠 Advanced AI Classification**: EfficientNet-B4 and MobileNetV3 models with >90% accuracy
- **👁️ Explainable AI**: GradCAM and LIME visualizations for transparent decision-making
- **✂️ Lesion Segmentation**: U-Net and V-Net architectures for precise boundary detection
- **🛡️ Risk Stratification**: Automated categorization into suspicious, non-suspicious, and high-risk classes
- **💻 Windows Desktop Tool**: Offline-capable application for rural and low-resource settings
- **📊 Real-time Monitoring**: AWS CloudWatch and Grafana dashboards
- **🏥 Clinical Integration**: Comprehensive reporting and patient management

## 🏗️ Architecture

### Web Application Stack
- **Backend**: Flask with SQLAlchemy
- **Frontend**: Modern HTML5/CSS3/JavaScript with Bootstrap 5
- **Database**: SQLite (development) / PostgreSQL (production)
- **AI Models**: PyTorch with segmentation-models-pytorch

### Desktop Application
- **Framework**: Tkinter with modern UI components
- **Packaging**: PyInstaller for Windows executable
- **Offline Capability**: Embedded AI models for rural deployment

## 🚀 Quick Start

### Web Application

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/oral-cancer-detection.git
   cd oral-cancer-detection/oralaid_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open http://localhost:5000 in your browser
   - Login with demo credentials: admin/admin123

### Desktop Application

1. **Run directly**
   ```bash
   python desktop_app.py
   ```

2. **Build Windows executable**
   ```bash
   python setup_windows.py
   ```

### Docker Deployment

1. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access services**
   - Web App: http://localhost:5000
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

## 📋 Clinical Classification System

| Provisional Diagnosis | AI Category | Risk Level |
|----------------------|-------------|------------|
| Oral cavity normal | Non-suspicious | Low-risk |
| Smokeless tobacco keratosis | Suspicious | Low-risk |
| Homogenous oral leukoplakia | Suspicious | Low-risk |
| Speckled oral leukoplakia | Suspicious | High-risk |
| Erythroplakia | Suspicious | High-risk |
| Proliferative verrucous leukoplakia | Suspicious | High-risk |
| Oral lichen planus | Suspicious | Low-risk |
| Squamous cell carcinoma | Suspicious | High-risk |

## 🔬 AI Models

### Classification Models
- **Primary**: EfficientNet-B4 (92.3% accuracy)
- **Fallback**: MobileNetV3-Large (lightweight deployment)
- **Training Dataset**: 608 patients with expert annotations

### Segmentation Models
- **Architecture**: U-Net with ResNet34 encoder
- **Purpose**: Precise lesion boundary detection
- **Output**: Binary masks for high-risk lesions

### Explainability Tools
- **GradCAM**: Visual attention heatmaps
- **LIME**: Local interpretable model explanations
- **Clinical Integration**: Transparent AI decision support

## 🖥️ User Interface

### Web Application Features
- **Modern Design**: Responsive Bootstrap 5 interface
- **Patient Management**: Comprehensive patient records
- **Real-time Analysis**: Progress indicators and live updates
- **Export Capabilities**: PDF reports and image downloads
- **Dashboard Analytics**: Usage statistics and trends

### Desktop Application Features
- **Offline Operation**: No internet required
- **Drag & Drop**: Intuitive image upload
- **Multi-tab Interface**: Original, GradCAM, and segmentation views
- **Export Options**: HTML reports and image sets
- **Clinical Recommendations**: Built-in guidance system

## 📊 Monitoring & Analytics

### Performance Metrics
- **Model Accuracy**: Real-time accuracy tracking
- **Response Times**: API and inference latency
- **Usage Statistics**: Patient volume and diagnosis trends
- **Error Rates**: Model confidence and failure analysis

### Grafana Dashboards
- **System Health**: Server resources and uptime
- **AI Performance**: Model accuracy and confidence scores
- **Clinical Metrics**: Risk distribution and case outcomes
- **User Activity**: Login patterns and feature usage

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=production
DATABASE_URL=sqlite:///oralaid.db
SECRET_KEY=your-secret-key
MODEL_PATH=models/
UPLOAD_FOLDER=uploads/
```

### Model Configuration
```python
# Model paths
CLASSIFIER_MODEL = "models/final_efficientnet_b4_oral_cancer.pth"
SEGMENTER_MODEL = "models/unet_segmenter.pth"

# Inference settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMAGE_SIZE = (224, 224)
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_models.py
pytest tests/test_api.py
pytest tests/test_ui.py
```

### Integration Tests
```bash
pytest tests/test_integration.py
```

### Performance Tests
```bash
python tests/benchmark_models.py
```

## 📦 Deployment

### Production Deployment
1. **AWS EC2 with Docker**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Traditional Server**
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
   ```

### Windows Executable Distribution
1. **Build executable**
   ```bash
   python setup_windows.py
   ```

2. **Create installer** (requires NSIS)
   ```bash
   makensis installer/installer.nsi
   ```

## 🔒 Security & Compliance

### Data Protection
- **HIPAA Compliance**: Patient data encryption and access controls
- **GDPR Ready**: Data anonymization and deletion capabilities
- **Audit Logging**: Comprehensive activity tracking
- **Secure Storage**: Encrypted database and file storage

### Model Security
- **Model Validation**: Checksum verification for AI models
- **Input Sanitization**: Image format and size validation
- **Rate Limiting**: API abuse prevention
- **Access Control**: Role-based permissions

## 📈 Performance Optimization

### Model Optimization
- **TensorRT**: GPU acceleration for NVIDIA cards
- **ONNX**: Cross-platform model optimization
- **Quantization**: Reduced model size for mobile deployment
- **Batch Processing**: Efficient multi-image analysis

### Web Optimization
- **CDN Integration**: Static asset delivery
- **Caching**: Redis for session and result caching
- **Compression**: Gzip for reduced bandwidth
- **Lazy Loading**: Progressive image loading

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run tests**
   ```bash
   pytest
   ```
5. **Submit pull request**

### Code Standards
- **PEP 8**: Python code formatting
- **Type Hints**: Function and variable annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Minimum 80% code coverage

## 📚 Documentation

### API Documentation
- **Swagger UI**: http://localhost:5000/api/docs
- **Postman Collection**: `docs/OralAID.postman_collection.json`
- **OpenAPI Spec**: `docs/openapi.yaml`

### Clinical Guidelines
- **User Manual**: `docs/clinical_user_manual.pdf`
- **Training Materials**: `docs/training/`
- **Validation Studies**: `docs/validation/`

## 🆘 Support

### Technical Support
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Community Forum**: User discussions and Q&A
- **Email Support**: technical-support@oralaid.com

### Clinical Support
- **Clinical Validation**: Peer-reviewed studies
- **Training Programs**: Webinars and workshops
- **Consultation**: Expert clinical guidance
- **Integration Support**: EHR and PACS integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Medical Device Disclaimer
This software is intended for research and educational purposes. It is not FDA-approved as a medical device. Clinical decisions should always be made by qualified healthcare professionals.

## 🙏 Acknowledgments

- **Clinical Partners**: Oral medicine specialists and pathologists
- **Research Institutions**: Academic collaborators
- **Open Source Community**: PyTorch, Flask, and other frameworks
- **Healthcare Organizations**: Early adopters and feedback providers

## 📞 Contact

- **Website**: https://oralaid.com
- **Email**: info@oralaid.com
- **GitHub**: https://github.com/oralaid/oralaid
- **LinkedIn**: https://linkedin.com/company/oralaid

---

**Built with ❤️ for better oral health outcomes worldwide**