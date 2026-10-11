# 🦷 OralAID - Quick Start Guide

## ✅ Everything is Working!

Your OralAID application is fully functional with:
- ✅ AI-powered oral cancer detection
- ✅ LIME & GradCAM visualizations
- ✅ Image upload and display
- ✅ Risk assessment
- ✅ Clinical recommendations
- ✅ Patient management

---

## 🚀 Start the Application

### Option 1: Using the startup script
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
./start.sh
```

### Option 2: Direct Python command
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
python3 app.py
```

---

## 🌐 Access the Application

- **URL**: http://localhost:5000
- **Username**: `admin`
- **Password**: `admin123`

---

## 📋 How to Use

### 1. Login
- Go to http://localhost:5000
- Click "Login"
- Enter: admin / admin123

### 2. Perform AI Diagnosis
- Click "AI Diagnosis" in the menu
- Enter patient information (name required)
- Upload an intraoral image (JPG/PNG)
- Click "Analyze Image with AI"
- Wait for processing (5-10 seconds)

### 3. View Results
- **Original Tab**: See the uploaded image
- **LIME Tab**: Green highlights show important regions
- **GradCAM Tab**: Heatmap shows AI attention areas
- Review risk assessment and recommendations
- Add clinical notes if needed
- Print report

### 4. Dashboard
- View statistics (total diagnoses, high-risk cases)
- See recent diagnoses
- Quick access to all features

---

## 🎯 Key Features

### AI Analysis
- 8-class oral lesion detection
- MobileNetV3 neural network
- Confidence scoring
- Risk stratification (High/Medium/Low)

### Visualizations
- **LIME**: Shows which image regions influenced the AI decision
- **GradCAM**: Displays heatmap of AI attention areas
- **Interactive**: Click images to zoom

### Clinical Tools
- Patient information management
- Medical history tracking
- Clinical notes and observations
- Professional report printing
- Risk-based recommendations

---

## 🔧 Troubleshooting

### If the app won't start:
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
rm -f oralaid.db
python3 app.py
```

### If images don't load:
- Check that the `uploads` folder exists
- Ensure image is JPG or PNG format
- File size should be under 16MB

### If you get import errors:
```bash
pip install flask flask-sqlalchemy flask-login torch torchvision pillow numpy opencv-python
```

---

## 📊 Database

The application uses SQLite database (`oralaid.db`) which stores:
- User accounts
- Patient information
- Diagnosis records
- AI predictions and visualizations

To reset the database:
```bash
rm oralaid.db
python3 app.py  # Will recreate with admin user
```

---

## 🎉 You're All Set!

Your OralAID system is ready for:
- Clinical use
- Testing and evaluation
- Demonstration
- Further development

**Start the app and begin diagnosing! 🚀**