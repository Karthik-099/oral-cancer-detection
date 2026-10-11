# OralAID Application - Run Commands

## Option 1: Flask Backend (Recommended)

### Step 1: Navigate to app directory
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
```

### Step 2: Activate virtual environment (if exists)
```bash
source ../oral/bin/activate
```

### Step 3: Run Flask application
```bash
python app.py
```

### Step 4: Access application
- Open browser: http://localhost:5000
- Login: admin / admin123

---

## Option 2: FastAPI Backend

### Step 1: Navigate to app directory
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
```

### Step 2: Activate virtual environment (if exists)
```bash
source ../oral/bin/activate
```

### Step 3: Run FastAPI application
```bash
python app_fastapi.py
```

### Step 4: Access application
- Open browser: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Or open: index_fastapi.html in browser

---

## Quick Start (Single Command)

### Flask:
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app && source ../oral/bin/activate && python app.py
```

### FastAPI:
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app && source ../oral/bin/activate && python app_fastapi.py
```

---

## Troubleshooting

### If virtual environment doesn't exist:
```bash
cd /home/karthik/oral-cancer-detection
python -m venv oral
source oral/bin/activate
pip install flask flask-sqlalchemy flask-login torch torchvision pillow opencv-python numpy
```

### If model file missing:
- Warning will show but app will run with pretrained weights
- Place trained model at: `/home/karthik/oral-cancer-detection/models/mobilenet_classifier.pth`
