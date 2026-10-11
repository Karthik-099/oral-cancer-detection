# 🚀 OralAID FastAPI - Complete Setup & Testing Guide

## 📋 Overview

This guide will help you:
1. Install FastAPI dependencies
2. Run the FastAPI server
3. Test all APIs
4. Connect frontend to backend
5. Deploy to production

---

## 1️⃣ Installation Steps

### Step 1: Install FastAPI Dependencies
```bash
cd /home/karthik/oral-cancer-detection/oralaid_app
pip3 install fastapi uvicorn[standard] python-multipart python-jose[cryptography] passlib[bcrypt]
```

### Step 2: Verify Dependencies
```bash
pip3 list | grep -E "fastapi|uvicorn|torch|opencv"
```

---

## 2️⃣ Running the Server

### Start FastAPI Server
```bash
python3 app_fastapi.py
```

**Expected Output:**
```
✅ AI Model loaded
✅ Created admin user (admin/admin123)
🚀 Starting OralAID FastAPI Server...
🌐 Access: http://localhost:8000
📚 API Docs: http://localhost:8000/docs
```

### Alternative: Run with Uvicorn
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3️⃣ Testing APIs

### Method 1: Swagger UI (Easiest)
1. Open browser: `http://localhost:8000/docs`
2. Click any endpoint
3. Click "Try it out"
4. Fill parameters
5. Click "Execute"

### Method 2: cURL Commands

#### Health Check
```bash
curl http://localhost:8000/api/health
```

#### Register User
```bash
curl -X POST "http://localhost:8000/api/register" \
  -F "username=doctor1" \
  -F "email=doctor1@test.com" \
  -F "password=test123" \
  -F "full_name=Dr. Smith"
```

#### Login
```bash
curl -X POST "http://localhost:8000/api/login" \
  -F "username=admin" \
  -F "password=admin123"
```

#### Predict (Replace with actual image path)
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@/path/to/oral_image.jpg" \
  -F "patient_name=John Doe" \
  -F "patient_age=45" \
  -F "patient_gender=Male"
```

#### Get Statistics
```bash
curl http://localhost:8000/api/stats
```

---

## 4️⃣ Frontend Connection

### Open Frontend in Browser
```bash
# Option 1: Direct file
firefox /home/karthik/oral-cancer-detection/oralaid_app/index_fastapi.html

# Option 2: Serve via HTTP (recommended)
cd /home/karthik/oral-cancer-detection/oralaid_app
python3 -m http.server 3000
# Then open: http://localhost:3000/index_fastapi.html
```

### Test Frontend Features
1. **Statistics** - Auto-loads on page open
2. **Upload Image** - Click upload area, select image
3. **Fill Patient Info** - Name, age, gender, history
4. **Analyze** - Click button, wait for results
5. **View Results** - Prediction, confidence, LIME, GradCAM

---

## 5️⃣ API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page with API info |
| `/docs` | GET | Interactive API documentation |
| `/api/health` | GET | Server health check |
| `/api/register` | POST | Register new user |
| `/api/login` | POST | User authentication |
| `/api/predict` | POST | AI image analysis |
| `/api/stats` | GET | System statistics |
| `/api/diagnoses` | GET | Recent diagnoses list |

---

## 6️⃣ Troubleshooting

### Port Already in Use
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn app_fastapi:app --port 8001
```

### Module Not Found
```bash
pip3 install --user fastapi uvicorn python-multipart
```

### CORS Error
- Already configured in app
- If persists, serve HTML via HTTP server (see step 4)

### Database Issues
```bash
# Reset database
rm oralaid_fastapi.db
python3 app_fastapi.py
```

---

## 7️⃣ Production Deployment

### Using Gunicorn
```bash
pip3 install gunicorn
gunicorn app_fastapi:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```bash
docker build -t oralaid-fastapi .
docker run -p 8000:8000 oralaid-fastapi
```

### Using Systemd Service
Create `/etc/systemd/system/oralaid.service`:
```ini
[Unit]
Description=OralAID FastAPI
After=network.target

[Service]
User=karthik
WorkingDirectory=/home/karthik/oral-cancer-detection/oralaid_app
ExecStart=/usr/bin/python3 app_fastapi.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable oralaid
sudo systemctl start oralaid
```

---

## 8️⃣ Testing Checklist

- [ ] Server starts without errors
- [ ] Health endpoint returns "healthy"
- [ ] Can register new user
- [ ] Can login with admin/admin123
- [ ] Can upload image and get prediction
- [ ] Statistics endpoint returns data
- [ ] Frontend loads and displays stats
- [ ] Frontend can upload and analyze images
- [ ] Results display correctly with visualizations
- [ ] Database stores records properly

---

## 📊 Expected API Responses

### Health Check Response
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "device": "cpu"
}
```

### Prediction Response
```json
{
  "success": true,
  "diagnosis_id": "abc-123",
  "patient_id": "PAT_20240125_xyz",
  "prediction": "Normal",
  "confidence": 95.67,
  "risk_level": "Low-risk",
  "risk_category": "Non-suspicious",
  "color": "#28a745",
  "lime_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,...",
  "image_url": "/uploads/filename.jpg"
}
```

---

## ✅ Summary

**What You Have:**
- ✅ Trained AI models (4 models in /models/)
- ✅ Complete Flask app (original)
- ✅ New FastAPI app (app_fastapi.py)
- ✅ Modern frontend (index_fastapi.html)
- ✅ 608 patient dataset
- ✅ All dependencies resolved

**What To Do:**
1. Install FastAPI: `pip3 install fastapi uvicorn python-multipart`
2. Run server: `python3 app_fastapi.py`
3. Test APIs: Open `http://localhost:8000/docs`
4. Test frontend: Open `index_fastapi.html` in browser
5. Verify everything works

**Key URLs:**
- API Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000/index_fastapi.html (via HTTP server)

---

**🎯 You're ready to go! Follow the steps above and test each component.**
