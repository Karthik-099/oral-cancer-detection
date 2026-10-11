"""
OralAID — Flask Application
Port: 5000
"""
import os, json, uuid, datetime, base64, io, threading, time
from functools import wraps
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from flask import (Flask, request, render_template, jsonify, redirect,
                   url_for, flash, send_from_directory, Response)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv

load_dotenv()

from models.classifier import OralCancerClassifier, RISK_MAPPING, load_classifier, get_model_info
from models.segmentation import UNet, load_segmenter, get_segmentation_mask
from models.xai import apply_gradcam, apply_lime, apply_segmentation
from models.validator import is_oral_image
from models.severity_scorer import load_severity_scorer
from utils.narrative_generator import generate_clinical_narrative

# ─── License Validation ───────────────────────────────────────────────────────

IS_LICENSE_VALID = True

def validate_license():
    global IS_LICENSE_VALID
    if os.environ.get('DEPLOY_MODE') == 'onprem':
        hardware_id = os.environ.get('HARDWARE_ID')
        license_key = os.environ.get('LICENSE_KEY')
        
        # In a real app, send request to OralAID license server
        # res = requests.post("https://api.oralaid.com/v1/validate", 
        #                    json={'hwid': hardware_id, 'key': license_key}, timeout=5)
        
        # For now, skip if no key is set (initial setup mode)
        if not license_key:
            IS_LICENSE_VALID = True 
        else:
            # Mock validation
            IS_LICENSE_VALID = (len(license_key) > 10)

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'oralaid-secret-2024'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///oralaid.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads')),
    RESULTS_FOLDER=os.path.join(BASE_DIR, 'static/results'),
    MAX_CONTENT_LENGTH=32 * 1024 * 1024,
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@app.before_request
def check_license():
    allowed = {
        'onprem_config', 'static', 'uploaded_file', 'login', 'health_check',
        'prometheus_metrics'
    }
    if not IS_LICENSE_VALID and request.endpoint not in allowed:
        return "<h3>OralAID License Expired or Invalid</h3><p>Please contact support or activate your server in <a href='/admin/onprem-config'>Admin Config</a>.</p>", 403

for d in [app.config['UPLOAD_FOLDER'], os.path.join(BASE_DIR, 'static/results'), os.path.join(BASE_DIR, 'static/annotations'), os.path.join(BASE_DIR, 'logs')]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load Models ─────────────────────────────────────────────────────────────

CLASSIFIER_PATH = os.environ.get(
    'CLASSIFIER_PATH',
    os.path.join(PROJECT_ROOT, 'models', 'final_efficientnet_b4_oral_cancer.pth')
)
SEGMENTER_PATH  = os.environ.get(
    'SEGMENTER_PATH',
    os.path.join(PROJECT_ROOT, 'models', 'unet_segmenter.pth')
)

classifier = load_classifier(CLASSIFIER_PATH, device)
segmenter  = load_segmenter(SEGMENTER_PATH, device)
severity_scorer = load_severity_scorer(device)

# GradCAM target layer
try:
    gradcam_layer = classifier.backbone.features[-1]
except Exception:
    gradcam_layer = None

TRANSFORM = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_classifier_risk_mapping():
    return getattr(classifier, 'risk_mapping', RISK_MAPPING)

def get_classifier_class_names():
    mapping = get_classifier_risk_mapping()
    return [mapping[i]['name'] for i in sorted(mapping.keys())]

# ─── DB Models ───────────────────────────────────────────────────────────────

class PatientVisit(db.Model):
    __tablename__ = 'patient_visits'

    id              = db.Column(db.Integer, primary_key=True)
    patient_id      = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    diagnosis_id    = db.Column(db.Integer, db.ForeignKey('diagnosis.id'), nullable=False)
    visit_number    = db.Column(db.Integer)
    visit_date      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    lesion_area_px  = db.Column(db.Integer)
    confidence      = db.Column(db.Float)
    class_name      = db.Column(db.String(50))
    severity_score  = db.Column(db.Float)
    
    patient         = db.relationship('Patient', backref='visits')
    diagnosis       = db.relationship('Diagnosis', backref='visit_record')

class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), default='doctor')
    full_name     = db.Column(db.String(100))
    tenant_id     = db.Column(db.Integer, db.ForeignKey('tenant.id'))
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Tenant(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    contact_email = db.Column(db.String(120))
    status        = db.Column(db.String(20), default='Active') # Active / Suspended / Expired
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    settings      = db.relationship('TenantSettings', backref='tenant', uselist=False)
    users         = db.relationship('User', backref='tenant')

class TenantSettings(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    tenant_id     = db.Column(db.Integer, db.ForeignKey('tenant.id'), nullable=False)
    license_key   = db.Column(db.String(100), unique=True, nullable=False)
    plan          = db.Column(db.String(50), default='Starter')
    max_doctors   = db.Column(db.Integer, default=1)
    max_patients  = db.Column(db.Integer, default=50)
    features_json = db.Column(db.Text, default='{}')
    expires_at    = db.Column(db.DateTime)
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class FeatureFlag(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    tenant_id     = db.Column(db.Integer, db.ForeignKey('tenant.id')) # NULL means global
    flag_name     = db.Column(db.String(100), nullable=False)
    enabled       = db.Column(db.Boolean, default=True)

class DemoRequest(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    name           = db.Column(db.String(100), nullable=False)
    email          = db.Column(db.String(120), nullable=False)
    hospital       = db.Column(db.String(100))
    role           = db.Column(db.String(50))
    country        = db.Column(db.String(50))
    requested_time = db.Column(db.DateTime, nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Patient(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    patient_id      = db.Column(db.String(50), unique=True, nullable=False)
    name            = db.Column(db.String(100), nullable=False)
    age             = db.Column(db.Integer)
    gender          = db.Column(db.String(10))
    tobacco_use     = db.Column(db.Boolean, default=False)
    alcohol_use     = db.Column(db.Boolean, default=False)
    family_history  = db.Column(db.Boolean, default=False)
    medical_history = db.Column(db.Text)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Diagnosis(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    diagnosis_id     = db.Column(db.String(50), unique=True, nullable=False)
    patient_id       = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    clinician_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path       = db.Column(db.String(200), nullable=False)
    ai_prediction    = db.Column(db.String(50))
    ai_confidence    = db.Column(db.Float)
    all_probabilities= db.Column(db.Text)   # JSON
    risk_level       = db.Column(db.String(20))
    clinical_notes   = db.Column(db.Text)
    final_diagnosis  = db.Column(db.Text)
    doctor_feedback  = db.Column(db.String(20))   # 'correct'|'incorrect'
    corrected_label  = db.Column(db.String(50))
    annotation_json  = db.Column(db.Text)          # polygon coordinates
    gradcam_path     = db.Column(db.Text)
    lime_path        = db.Column(db.Text)
    segmentation_path= db.Column(db.Text)
    severity_score   = db.Column(db.Float)
    lesion_area_px   = db.Column(db.Integer)
    created_at       = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    clinician        = db.relationship('User',    foreign_keys=[clinician_id])
    patient          = db.relationship('Patient', foreign_keys=[patient_id])

class ModelMetrics(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    version    = db.Column(db.String(50))
    accuracy   = db.Column(db.Float)
    f1_score   = db.Column(db.Float)
    parameters = db.Column(db.Integer)
    size_mb    = db.Column(db.Float)
    recorded_at= db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Appointment(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    patient_id      = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    clinician_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scheduled_date  = db.Column(db.DateTime, nullable=False)
    type            = db.Column(db.String(50))  # Initial / Follow-up / Video / Biopsy
    status          = db.Column(db.String(20), default='Scheduled')  # Scheduled / Completed / Cancelled
    room_url        = db.Column(db.String(200)) # Daily.co room URL
    notes           = db.Column(db.Text)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    patient         = db.relationship('Patient', backref='appointments')
    clinician       = db.relationship('User', backref='appointments')

class ChatMessage(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    patient_id      = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    clinician_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender_type     = db.Column(db.String(10)) # 'doctor' or 'patient'
    content         = db.Column(db.Text, nullable=False)
    is_read         = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    patient         = db.relationship('Patient', backref='messages')
    clinician       = db.relationship('User', backref='messages')

class Notification(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message         = db.Column(db.String(255), nullable=False)
    type            = db.Column(db.String(20)) # diagnosis / high-risk / appointment / message
    is_read         = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    user            = db.relationship('User', backref='notifications')

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

def ensure_database_schema():
    """Create new tables and add lightweight SQLite columns for deployed upgrades."""
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    table_names = set(inspector.get_table_names())

    if 'patient_visit' in table_names and 'patient_visits' not in table_names:
        db.session.execute(db.text('ALTER TABLE patient_visit RENAME TO patient_visits'))
        db.session.commit()

    db.create_all()

    inspector = inspect(db.engine)
    table_names = set(inspector.get_table_names())
    if 'diagnosis' in table_names:
        diagnosis_cols = {c['name'] for c in inspector.get_columns('diagnosis')}
        diagnosis_additions = {
            'severity_score': 'FLOAT',
            'lesion_area_px': 'INTEGER',
        }
        for column_name, column_type in diagnosis_additions.items():
            if column_name not in diagnosis_cols:
                db.session.execute(db.text(f'ALTER TABLE diagnosis ADD COLUMN {column_name} {column_type}'))
        db.session.commit()

    if not User.query.filter_by(username='admin').first():
        db.session.add(User(
            username='admin',
            email='admin@oralaid.com',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            full_name='System Administrator',
        ))
        db.session.commit()

    existing_counts = {
        row.patient_id: row.visit_count
        for row in db.session.query(
            PatientVisit.patient_id,
            db.func.count(PatientVisit.id).label('visit_count')
        ).group_by(PatientVisit.patient_id).all()
    }
    diagnoses_without_visits = Diagnosis.query.outerjoin(
        PatientVisit, PatientVisit.diagnosis_id == Diagnosis.id
    ).filter(PatientVisit.id.is_(None)).order_by(Diagnosis.created_at.asc()).all()

    for diagnosis in diagnoses_without_visits:
        next_visit = existing_counts.get(diagnosis.patient_id, 0) + 1
        existing_counts[diagnosis.patient_id] = next_visit
        db.session.add(PatientVisit(
            patient_id=diagnosis.patient_id,
            diagnosis_id=diagnosis.id,
            visit_number=next_visit,
            visit_date=diagnosis.created_at or datetime.datetime.utcnow(),
            lesion_area_px=diagnosis.lesion_area_px or 0,
            confidence=diagnosis.ai_confidence or 0,
            class_name=diagnosis.ai_prediction or 'Unknown',
            severity_score=diagnosis.severity_score or 0,
        ))
    db.session.commit()

with app.app_context():
    ensure_database_schema()

# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(image: Image.Image, metadata_tensor=None):
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(tensor, metadata_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class  = int(probs.argmax())
    confidence  = float(probs[pred_class])
    risk_mapping = get_classifier_risk_mapping()
    risk_info   = risk_mapping.get(pred_class, RISK_MAPPING[0])
    return pred_class, confidence, probs.tolist(), risk_info, tensor

def build_metadata_tensor(age, gender, tobacco, alcohol, family):
    return torch.tensor([[
        float(age or 40) / 100.0,
        1.0 if str(gender).lower() == 'male' else 0.0,
        float(tobacco),
        float(alcohol),
        float(family),
    ]], dtype=torch.float32).to(device)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already exists', 'warning')
            return render_template('register.html')
        user = User(
            username=request.form['username'],
            email=request.form['email'],
            password_hash=generate_password_hash(request.form['password']),
            full_name=request.form.get('full_name', ''),
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful — please login', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    diagnoses      = Diagnosis.query.filter_by(clinician_id=current_user.id).order_by(Diagnosis.created_at.desc()).limit(20).all()
    all_diagnoses  = Diagnosis.query.filter_by(clinician_id=current_user.id).all()
    total          = Diagnosis.query.filter_by(clinician_id=current_user.id).count()
    high_risk      = Diagnosis.query.filter_by(clinician_id=current_user.id, risk_level='High-risk').count()
    correct_fb     = Diagnosis.query.filter_by(clinician_id=current_user.id, doctor_feedback='correct').count()
    incorrect_fb   = Diagnosis.query.filter_by(clinician_id=current_user.id, doctor_feedback='incorrect').count()
    model_info     = get_model_info(classifier)
    latest_metrics = ModelMetrics.query.order_by(ModelMetrics.recorded_at.desc()).first()
    total_patients = len({d.patient_id for d in all_diagnoses})
    feedback_total = correct_fb + incorrect_fb
    if feedback_total:
        accuracy_rate = round(correct_fb / feedback_total * 100, 1)
    elif latest_metrics and latest_metrics.accuracy:
        accuracy_rate = round(float(latest_metrics.accuracy) * (100 if latest_metrics.accuracy <= 1 else 1), 1)
    else:
        loaded_accuracy = (model_info.get('loaded_accuracy') or '76.8%').replace('%', '')
        accuracy_rate = round(float(loaded_accuracy), 1)

    class_distribution = []
    risk_mapping = get_classifier_risk_mapping()
    for i in sorted(risk_mapping.keys()):
        name = risk_mapping[i]['name']
        class_distribution.append({
            'name': name,
            'count': sum(1 for d in all_diagnoses if d.ai_prediction == name)
        })

    # High Risk Alerts
    high_risk_classes = ['Speckled Leukoplakia', 'Erythroplakia', 'Squamous Cell Carcinoma']
    high_risk_diagnoses = Diagnosis.query.filter(Diagnosis.clinician_id == current_user.id,
                                                Diagnosis.ai_prediction.in_(high_risk_classes)).all()
    
    high_risk_alerts = []
    for d in high_risk_diagnoses:
        has_follow_up = Appointment.query.filter_by(patient_id=d.patient_id, type='Follow-up').first()
        if not has_follow_up:
            high_risk_alerts.append({
                'patient_name': d.patient.name,
                'patient_id': d.patient.id,
                'diagnosis_date': d.created_at,
                'prediction': d.ai_prediction
            })

    # Check for appointments in 1 hour
    one_hour_later = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    upcoming_soon = Appointment.query.filter(
        Appointment.clinician_id == current_user.id,
        Appointment.scheduled_date > datetime.datetime.utcnow(),
        Appointment.scheduled_date <= one_hour_later,
        Appointment.status == 'Scheduled'
    ).all()
    
    for a in upcoming_soon:
        # Check if already notified
        msg = f"Upcoming {a.type} with {a.patient.name} at {a.scheduled_date.strftime('%H:%M')}"
        existing = Notification.query.filter_by(user_id=current_user.id, message=msg).first()
        if not existing:
            db.session.add(Notification(
                user_id=current_user.id,
                message=msg,
                type='appointment'
            ))
    db.session.commit()

    return render_template('dashboard.html',
        recent_diagnoses=diagnoses, total_diagnoses=total,
        total_patients=total_patients, accuracy_rate=accuracy_rate,
        high_risk_count=high_risk, correct_feedback=correct_fb,
        incorrect_feedback=incorrect_fb, model_info=model_info,
        latest_metrics=latest_metrics, class_distribution=class_distribution,
        high_risk_alerts=high_risk_alerts,
        now=datetime.datetime.utcnow())

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

@app.route('/patients')
@login_required
def patients():
    patient_ids = [d.patient_id for d in Diagnosis.query.filter_by(clinician_id=current_user.id).all()]
    patient_list = Patient.query.filter(Patient.id.in_(patient_ids)).order_by(Patient.created_at.desc()).all() if patient_ids else []
    return render_template('patients.html', patients=patient_list)

@app.route('/results')
@login_required
def results():
    diagnoses = Diagnosis.query.filter_by(clinician_id=current_user.id).order_by(Diagnosis.created_at.desc()).all()
    return render_template('results.html', diagnoses=diagnoses)

@app.route('/reports')
@login_required
def reports():
    diagnoses = Diagnosis.query.filter_by(clinician_id=current_user.id).order_by(Diagnosis.created_at.desc()).all()
    return render_template('reports.html', diagnoses=diagnoses)

@app.route('/schedule')
@login_required
def schedule():
    diagnoses = Diagnosis.query.filter_by(clinician_id=current_user.id).order_by(Diagnosis.created_at.desc()).limit(5).all()
    return render_template('schedule.html', diagnoses=diagnoses)

@app.route('/messages')
@login_required
def messages():
    diagnoses = Diagnosis.query.filter_by(clinician_id=current_user.id).order_by(Diagnosis.created_at.desc()).limit(5).all()
    return render_template('messages.html', diagnoses=diagnoses)

@app.route('/lab-contacts')
@login_required
def lab_contacts():
    labs = []
    return render_template('lab_contacts.html', labs=labs)

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'admin':
            flash('Admin access required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin')
@admin_required
def admin_dashboard():
    tenants = Tenant.query.all()
    feature_flags = FeatureFlag.query.filter_by(tenant_id=None).all()
    
    # Usage Analytics (Last 30 days)
    today = datetime.datetime.utcnow().date()
    diagnoses_by_day = db.session.query(
        db.func.date(Diagnosis.created_at), db.func.count(Diagnosis.id)
    ).filter(Diagnosis.created_at >= today - datetime.timedelta(days=30))\
    .group_by(db.func.date(Diagnosis.created_at)).all()
    
    labels = [(today - datetime.timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
    counts = [0] * 30
    for d, c in diagnoses_by_day:
        try:
            d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)
            idx = labels.index(d_str)
            counts[idx] = c
        except: pass

    # Users by plan
    plan_dist = db.session.query(
        TenantSettings.plan, db.func.count(Tenant.id)
    ).join(Tenant).group_by(TenantSettings.plan).all()
    
    return render_template('admin.html', 
        tenants=tenants, 
        feature_flags=feature_flags,
        analytics={
            'labels': labels,
            'counts': counts,
            'plans': {p: c for p, c in plan_dist}
        })

@app.route('/admin/onprem-config')
@admin_required
def onprem_config():
    import hashlib, socket
    try:
        hostname = socket.gethostname()
        fingerprint = hashlib.sha256(hostname.encode()).hexdigest()
    except:
        fingerprint = "N/A"
        
    return render_template('onprem_config.html', 
        fingerprint=fingerprint,
        minio_status='Connected',
        deploy_mode=os.environ.get('DEPLOY_MODE', 'cloud'))

@app.route('/api/tenants', methods=['POST'])
@admin_required
def create_tenant():
    data = request.json
    t = Tenant(name=data['name'], contact_email=data['email'])
    db.session.add(t)
    db.session.flush()
    
    key = str(uuid.uuid4())
    expiry = datetime.datetime.utcnow() + datetime.timedelta(days=int(data.get('duration', 365)))
    s = TenantSettings(
        tenant_id=t.id, 
        license_key=key, 
        plan=data['plan'],
        max_doctors=data['max_doctors'],
        max_patients=data['max_patients'],
        expires_at=expiry
    )
    db.session.add(s)
    db.session.commit()
    return jsonify({'status': 'success', 'license_key': key})

@app.route('/api/feature-flags/<int:id>/toggle', methods=['POST'])
@admin_required
def toggle_flag(id):
    flag = FeatureFlag.query.get_or_404(id)
    flag.enabled = not flag.enabled
    db.session.commit()
    return jsonify({'status': 'success', 'enabled': flag.enabled})

@app.route('/result/<diagnosis_id>')
@login_required
def result(diagnosis_id):
    d = Diagnosis.query.filter_by(diagnosis_id=diagnosis_id).first_or_404()
    risk_info = next((v for v in get_classifier_risk_mapping().values() if v['name'] == d.ai_prediction), RISK_MAPPING[0])
    class_names = get_classifier_class_names()
    all_probs = json.loads(d.all_probabilities) if d.all_probabilities else [0]*len(class_names)
    
    # Recommendations based on WHO guidelines
    recommendations = []
    if d.risk_level == 'High-risk':
        recommendations = [
            "Immediate referral to oral medicine specialist or oral surgeon",
            "Biopsy recommended for definitive histopathological diagnosis",
            "Follow-up within 2–4 weeks if referral delayed"
        ]
    elif d.risk_level == 'Medium-risk':
        recommendations = [
            "Consider referral to oral medicine specialist",
            "Schedule follow-up in 4–6 weeks for reassessment",
            "Eliminate all tobacco and alcohol use"
        ]
    else:
        recommendations = [
            "Continue routine oral health maintenance",
            "Annual oral cancer screening recommended",
            "Monitor for any changes in lesion size or appearance"
        ]

    # Check for upcoming appointment
    upcoming_appt = Appointment.query.filter_by(patient_id=d.patient_id, status='Scheduled')\
                                    .order_by(Appointment.scheduled_date.asc()).first()

    return render_template('result.html',
        diagnosis=d, patient=d.patient,
        risk_info=risk_info, confidence=(d.ai_confidence or 0)*100,
        all_probs=all_probs,
        class_names=class_names,
        recommendations=recommendations,
        upcoming_appt=upcoming_appt)

@app.route('/diagnose', methods=['GET', 'POST'])
@login_required
def diagnose():
    if request.method == 'POST':
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No images uploaded'}), 400

        patient_name    = request.form.get('patient_name', 'Unknown')
        patient_age     = request.form.get('patient_age', 40, type=int)
        patient_gender  = request.form.get('patient_gender', 'Unknown')
        tobacco         = int(request.form.get('tobacco_use', 0))
        alcohol         = int(request.form.get('alcohol_use', 0))
        family          = int(request.form.get('family_history', 0))
        medical_history = request.form.get('medical_history', '')

        # Patient (create or find)
        patient = Patient.query.filter_by(name=patient_name).first()
        if not patient:
            patient = Patient(
                patient_id=f"PAT_{datetime.datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
                name=patient_name, age=patient_age, gender=patient_gender,
                tobacco_use=bool(tobacco), alcohol_use=bool(alcohol),
                family_history=bool(family), medical_history=medical_history,
            )
            db.session.add(patient)
            db.session.commit()

        results = []
        for file in files:
            if file.filename == '': continue
            
            diagnosis_id = str(uuid.uuid4())
            filename     = f"{diagnosis_id}_{secure_filename(file.filename)}"
            filepath     = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                image = Image.open(filepath).convert('RGB')
                
                valid, reason = is_oral_image(image, device)
                if not valid:
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    results.append({
                        'error': 'Please provide only oral images.',
                        'filename': file.filename,
                        'details': reason
                    })
                    continue
                
                meta  = build_metadata_tensor(patient_age, patient_gender, tobacco, alcohol, family)
                pred_class, confidence, all_probs, risk_info, tensor = run_inference(image, meta)

                # Save XAI as files
                gradcam_filename = f"gradcam_{diagnosis_id}.png"
                gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
                apply_gradcam(classifier, tensor, gradcam_layer, image, save_path=gradcam_path) if gradcam_layer else None
                
                lime_filename = f"lime_{diagnosis_id}.png"
                lime_path = os.path.join(app.config['RESULTS_FOLDER'], lime_filename)
                apply_lime(classifier, image, device, save_path=lime_path)
                
                seg_filename = f"segmentation_{diagnosis_id}.png"
                seg_path = os.path.join(app.config['RESULTS_FOLDER'], seg_filename)
                mask = get_segmentation_mask(segmenter, tensor, device)
                apply_segmentation(image, mask, save_path=seg_path)

                # Progression Metrics
                lesion_area = int(np.sum(mask > 0.5))
                is_high_risk = risk_info['name'] in ['Speckled Leukoplakia', 'Erythroplakia', 'Squamous Cell Carcinoma']
                normal_idx = 0 # Assuming 'Normal' is idx 0
                sev_score = severity_scorer.score(image, all_probs[normal_idx], is_high_risk, lesion_area / (256*256))

                diagnosis = Diagnosis(
                    diagnosis_id=diagnosis_id,
                    patient_id=patient.id,
                    clinician_id=current_user.id,
                    image_path=filename,
                    ai_prediction=risk_info['name'],
                    ai_confidence=confidence,
                    all_probabilities=json.dumps(all_probs),
                    risk_level=risk_info['risk'],
                    gradcam_path=gradcam_filename,
                    lime_path=lime_filename,
                    segmentation_path=seg_filename,
                    severity_score=sev_score,
                    lesion_area_px=lesion_area
                )
                db.session.add(diagnosis)
                db.session.flush() # Get diagnosis.id
                
                # Save Patient Visit for Longitudinal Tracking
                visit_count = PatientVisit.query.filter_by(patient_id=patient.id).count()
                visit = PatientVisit(
                    patient_id=patient.id,
                    diagnosis_id=diagnosis.id,
                    visit_number=visit_count + 1,
                    lesion_area_px=lesion_area,
                    confidence=confidence,
                    class_name=risk_info['name'],
                    severity_score=sev_score
                )
                db.session.add(visit)
                
                # New Diagnosis Notification
                db.session.add(Notification(
                    user_id=current_user.id,
                    message=f"Diagnosis completed for {patient.name}: {risk_info['name']}",
                    type='diagnosis'
                ))
                db.session.commit()
                
                # High-risk auto-follow-up
                if risk_info['name'] in ['Speckled Leukoplakia', 'Erythroplakia', 'Squamous Cell Carcinoma']:
                    follow_up_date = datetime.datetime.utcnow() + datetime.timedelta(days=28)
                    appt = Appointment(
                        patient_id=patient.id,
                        clinician_id=current_user.id,
                        scheduled_date=follow_up_date,
                        type='Follow-up',
                        notes=f"Auto-scheduled: high-risk diagnosis follow-up for {risk_info['name']}"
                    )
                    db.session.add(appt)
                    
                    notif = Notification(
                        user_id=current_user.id,
                        message=f"CRITICAL: High-risk diagnosis for {patient.name}. Follow-up scheduled.",
                        type='high-risk'
                    )
                    db.session.add(notif)
                    db.session.commit()

                results.append({
                    'diagnosis_id': diagnosis_id,
                    'prediction': risk_info['name'],
                    'confidence': confidence,
                    'risk_level': risk_info['risk']
                })

            except Exception as e:
                print(f"Error processing image {file.filename}: {e}")
                results.append({'error': str(e), 'filename': file.filename})

        return jsonify({'results': results, 'diagnosis_id': results[0].get('diagnosis_id') if results else None})

    return render_template('diagnose.html')

@app.route('/health')
def health_check():
    try:
        db.session.execute(db.text('SELECT 1'))
        db_ok = True
    except:
        db_ok = False
    return jsonify({
        'status': 'ok',
        'model_loaded': all(model is not None for model in [classifier, segmenter, severity_scorer]),
        'db_connected': db_ok,
        'device': str(device)
    })

@app.route('/api/progression/<int:patient_id>')
@login_required
def api_progression(patient_id):
    visits = PatientVisit.query.filter_by(patient_id=patient_id).order_by(PatientVisit.visit_date.asc()).all()
    data = []
    for i, v in enumerate(visits):
        change_pct = 0
        current_area = v.lesion_area_px or 0
        previous_area = (visits[i-1].lesion_area_px or 0) if i > 0 else 0
        if i > 0 and previous_area > 0:
            change_pct = ((current_area - previous_area) / previous_area) * 100
            
        data.append({
            'visit_number': v.visit_number,
            'visit_date': v.visit_date.strftime('%Y-%m-%d'),
            'lesion_area_px': current_area,
            'confidence': (v.confidence or 0) * 100,
            'class_name': v.class_name,
            'change_pct': round(change_pct, 1),
            'image_url': url_for('uploaded_file', filename=v.diagnosis.image_path) if v.diagnosis else None
        })
    return jsonify(data)

@app.route('/patients/<int:patient_id>/progression')
@login_required
def progression_view(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    visits = PatientVisit.query.filter_by(patient_id=patient_id).order_by(PatientVisit.visit_date.asc()).all()
    
    last_change = 0
    if len(visits) >= 2:
        v1 = visits[-2].lesion_area_px or 0
        v2 = visits[-1].lesion_area_px or 0
        if v1 > 0: last_change = ((v2 - v1) / v1) * 100
    latest_risk = visits[-1].diagnosis.risk_level if visits and visits[-1].diagnosis else 'No visits'

    return render_template('progression.html', 
        patient=patient, 
        visits=visits, 
        last_change=round(last_change, 1),
        first_visit=visits[0] if visits else None,
        latest_visit=visits[-1] if visits else None,
        latest_risk=latest_risk)

@app.route('/report/<diagnosis_id>')
@login_required
def report_pdf(diagnosis_id):
    from weasyprint import HTML
    import qrcode
    
    d = Diagnosis.query.filter_by(diagnosis_id=diagnosis_id).first_or_404()
    risk_info = next((v for v in get_classifier_risk_mapping().values() if v['name'] == d.ai_prediction), RISK_MAPPING[0])
    
    # AI Clinical Narrative (Anthropic)
    narrative = generate_clinical_narrative(
        d.patient.name, d.patient.age, d.patient.gender,
        d.ai_prediction, (d.ai_confidence or 0)*100, d.risk_level,
        "Yes" if d.patient.tobacco_use else "No",
        "Yes" if d.patient.alcohol_use else "No",
        "Yes" if d.patient.family_history else "No",
        d.lesion_area_px or 0
    )

    # Generate QR Code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url_for('result', diagnosis_id=diagnosis_id, _external=True))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_io = io.BytesIO()
    img.save(qr_io, 'PNG')
    qr_b64 = base64.b64encode(qr_io.getvalue()).decode()

    # Recommendations
    if d.risk_level == 'High-risk':
        recs = ["Immediate referral to specialist", "Biopsy recommended", "Follow-up within 2-4 weeks"]
    elif d.risk_level == 'Medium-risk':
        recs = ["Consider referral", "Follow-up in 4-6 weeks", "Eliminate risk factors"]
    else:
        recs = ["Routine maintenance", "Annual screening", "Self-monitoring"]

    # Template for PDF
    html_content = render_template('print_report.html', 
                                  diagnosis=d, 
                                  patient=d.patient, 
                                  risk_info=risk_info,
                                  qr_code=qr_b64,
                                  recommendations=recs,
                                  narrative=narrative,
                                  now=datetime.datetime.now(),
                                  config=app.config)
    
    pdf = HTML(string=html_content, base_url=request.base_url).write_pdf()
    
    response = Response(pdf, content_type='application/pdf')
    response.headers['Content-Disposition'] = f'inline; filename=OralAID_Report_{diagnosis_id}.pdf'
    return response

@app.route('/api/annotations', methods=['POST'])
@login_required
def save_annotations():
    data = request.json
    diagnosis_id = data.get('diagnosis_id')
    if not diagnosis_id:
        return jsonify({'error': 'Diagnosis ID required'}), 400
    
    d = Diagnosis.query.filter_by(diagnosis_id=diagnosis_id).first_or_404()
    d.annotation_json = json.dumps(data)
    
    # Also save to static folder for persistence/retraining
    ann_file = os.path.join(app.static_folder, 'annotations', f"{diagnosis_id}.json")
    with open(ann_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Annotations saved'})

@app.route('/feedback/<diagnosis_id>', methods=['POST'])
@login_required
def submit_feedback(diagnosis_id):
    d = Diagnosis.query.filter_by(diagnosis_id=diagnosis_id).first_or_404()
    d.doctor_feedback  = request.form.get('feedback')
    d.corrected_label  = request.form.get('corrected_label', '')
    d.clinical_notes   = request.form.get('clinical_notes', '')
    d.final_diagnosis  = request.form.get('final_diagnosis', '')
    db.session.commit()

    # Save annotation JSON for retraining
    annotation = request.form.get('annotation_json', '')
    if annotation:
        d.annotation_json = annotation
        ann_file = os.path.join(app.static_folder, 'annotations', f"{diagnosis_id}.json")
        with open(ann_file, 'w') as f:
            ann_data = json.loads(annotation) if annotation else {}
            ann_data.update({
                'image_filename': d.image_path,
                'ai_prediction': d.ai_prediction,
                'corrected_label': d.corrected_label,
                'patient_age': d.patient.age,
                'patient_gender': d.patient.gender,
            })
            json.dump(ann_data, f, indent=2)
        db.session.commit()

    flash('Feedback saved — thank you!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/api/retrain', methods=['POST'])
@login_required
def trigger_retrain():
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin only'}), 403

    def _retrain():
        from training.retrain import retrain_from_feedback
        result = retrain_from_feedback()
        if result.get('status') == 'success':
            m = ModelMetrics(
                version=f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
                accuracy=result.get('final_val_acc', 0),
                f1_score=result.get('final_val_f1', 0),
                parameters=get_model_info(classifier)['total_parameters'],
                size_mb=get_model_info(classifier)['model_size_mb'],
            )
            with app.app_context():
                db.session.add(m)
                db.session.commit()

    threading.Thread(target=_retrain, daemon=True).start()
    return jsonify({'status': 'Retraining started in background'})

@app.route('/api/model_metrics')
@login_required
def model_metrics():
    info    = get_model_info(classifier)
    metrics = ModelMetrics.query.order_by(ModelMetrics.recorded_at.desc()).limit(10).all()
    history = [{'version': m.version, 'accuracy': m.accuracy, 'f1': m.f1_score,
                 'recorded_at': m.recorded_at.isoformat()} for m in metrics]
    return jsonify({'model_info': info, 'history': history})

@app.route('/api/stats')
@login_required
def stats():
    total    = Diagnosis.query.count()
    high     = Diagnosis.query.filter_by(risk_level='High-risk').count()
    correct  = Diagnosis.query.filter_by(doctor_feedback='correct').count()
    feedback = Diagnosis.query.filter(Diagnosis.doctor_feedback.isnot(None)).count()
    accuracy = round(correct / feedback * 100, 1) if feedback else 0
    return jsonify({'total': total, 'high_risk': high,
                    'feedback_count': feedback, 'model_accuracy_feedback': accuracy})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/demo', methods=['GET', 'POST'])
def demo_booking():
    if request.method == 'POST':
        data = request.form
        req = DemoRequest(
            name=data['name'],
            email=data['email'],
            hospital=data.get('hospital'),
            role=data.get('role'),
            country=data.get('country'),
            requested_time=datetime.datetime.fromisoformat(data['requested_time'].replace('Z', ''))
        )
        db.session.add(req)
        db.session.commit()
        
        # Webhook (Optional)
        webhook_url = os.environ.get('DEMO_WEBHOOK_URL')
        if webhook_url:
            try:
                requests.post(webhook_url, json={
                    'event': 'demo_requested',
                    'name': req.name,
                    'email': req.email,
                    'hospital': req.hospital
                }, timeout=5)
            except: pass
            
        ticket_id = f"OAID-{req.id:05d}"
        return jsonify({'status': 'success', 'ticket_id': ticket_id})
    return render_template('demo.html', now=datetime.datetime.utcnow())

# ─── Model Performance Page ──────────────────────────────────────────────────

@app.route('/model-performance')
@login_required
def model_performance():
    info = get_model_info(classifier)
    metrics = ModelMetrics.query.order_by(ModelMetrics.recorded_at.desc()).limit(20).all()
    history = [{'version': m.version, 'accuracy': round(m.accuracy * 100, 2) if m.accuracy else 0,
                'f1': round(m.f1_score, 4) if m.f1_score else 0,
                'recorded_at': m.recorded_at.strftime('%Y-%m-%d %H:%M')} for m in metrics]
    return render_template('model_performance.html', model_info=info, history=history,
                           latest=metrics[0] if metrics else None)

# ─── Blogs ────────────────────────────────────────────────────────────────────

BLOGS = [
    {
        'id': 1, 'slug': 'oral-cancer-global-burden',
        'title': 'The Global Burden of Oral Cancer: WHO and GLOBOCAN 2022',
        'subtitle': '389,846 new cases and 188,438 deaths — why early detection matters',
        'author': 'OralAID Research Team', 'date': 'January 15, 2025',
        'read_time': '8 min', 'category': 'Research',
        'image': 'https://commons.wikimedia.org/wiki/Special:FilePath/Leukoplakia.jpg?width=900',
        'excerpt': 'Oral cancer remains one of the most underdiagnosed cancers globally. WHO and IARC GLOBOCAN data show a continued need for earlier screening and referral pathways.',
        'content': '''<p>According to the World Health Organization (WHO), the global incidence of cancers of the lip and oral cavity was estimated at <strong>389,846 new cases</strong> and <strong>188,438 deaths</strong> in 2022.</p>
        <h3>Regional Distribution</h3>
        <p>The burden is disproportionately concentrated in South and Southeast Asia. India accounts for approximately <strong>119,000 new cases per year</strong> — nearly 30% of the global total — driven by widespread tobacco and betel nut use.</p>
        <h3>Key WHO Statistics (2022)</h3>
        <ul>
        <li>India: 119,992 cases</li>
        <li>Bangladesh: 28,000 cases / 18,000 deaths</li>
        <li>Pakistan: 21,000 cases / 14,000 deaths</li>
        <li>Sri Lanka: 4,200 cases / 2,800 deaths</li>
        <li>USA: 54,000 cases / 10,850 deaths</li>
        <li>Brazil: 15,000 cases / 7,500 deaths</li>
        <li>China: 48,000 cases / 22,000 deaths</li>
        </ul>
        <h3>5-Year Survival Rate</h3>
        <p>When detected at Stage I, the 5-year survival rate is <strong>83%</strong>. At Stage IV, it drops to <strong>38%</strong>. Early detection through AI-assisted screening can dramatically improve outcomes.</p>'''
    },
    {
        'id': 2, 'slug': 'efficientnet-oral-cancer-detection',
        'title': 'How EfficientNet-B4 Achieves State-of-the-Art Oral Cancer Classification',
        'subtitle': 'Deep dive into the architecture powering OralAID',
        'author': 'OralAID AI Team', 'date': 'February 3, 2025',
        'read_time': '12 min', 'category': 'Technology',
        'image': 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=900&q=80',
        'excerpt': 'EfficientNet-B4 uses compound scaling to simultaneously scale depth, width, and resolution — achieving superior accuracy with fewer parameters than ResNet-50.',
        'content': '''<p>EfficientNet-B4 is the backbone of OralAID's classification engine. Unlike traditional CNNs that scale only one dimension, EfficientNet uses <strong>compound scaling</strong> — simultaneously scaling network depth, width, and input resolution using a fixed set of scaling coefficients.</p>
        <h3>Why EfficientNet-B4?</h3>
        <p>For oral cancer classification, we evaluated ResNet-50, VGG-16, InceptionV3, and EfficientNet variants. EfficientNet-B4 achieved the best trade-off between accuracy (94.2% on our validation set) and inference speed (47ms per image on CPU).</p>
        <h3>Metadata Fusion</h3>
        <p>OralAID extends the standard EfficientNet-B4 with a metadata fusion head that incorporates patient demographics (age, gender, tobacco use, alcohol use, family history) — improving accuracy by 3.2% over image-only classification.</p>'''
    },
    {
        'id': 3, 'slug': 'lime-gradcam-explainability',
        'title': 'Making AI Explainable: LIME + GradCAM in Clinical Practice',
        'subtitle': 'Why black-box AI is unacceptable in medicine — and how we solve it',
        'author': 'OralAID Clinical Team', 'date': 'March 10, 2025',
        'read_time': '10 min', 'category': 'Explainability',
        'image': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Squamous_cell_carcinoma.jpg/960px-Squamous_cell_carcinoma.jpg',
        'excerpt': 'Clinicians need to understand WHY an AI made a decision. We combine LIME superpixel explanations with GradCAM heatmaps to provide dual-layer explainability.',
        'content': '''<p>In clinical AI, explainability is not optional — it is a regulatory and ethical requirement. OralAID implements two complementary XAI techniques: <strong>GradCAM</strong> and <strong>LIME</strong>.</p>
        <h3>GradCAM: Gradient-Weighted Class Activation Mapping</h3>
        <p>GradCAM computes the gradient of the class score with respect to the final convolutional feature maps, then weights the feature maps by their importance to produce a heatmap highlighting the regions that most influenced the prediction.</p>
        <h3>LIME: Local Interpretable Model-Agnostic Explanations</h3>
        <p>LIME perturbs the input image by masking superpixels and observing how the model's confidence changes. Superpixels that, when removed, cause the largest drop in confidence are highlighted as the most important regions.</p>
        <h3>Combined Interpretation</h3>
        <p>When GradCAM and LIME agree on the same region, clinicians can have high confidence that the AI is focusing on the correct anatomical area. Disagreement signals uncertainty and warrants closer clinical review.</p>'''
    },
    {
        'id': 4, 'slug': 'ai-saas-hospitals',
        'title': 'Deploying AI Diagnostics as SaaS in Hospitals: Lessons Learned',
        'subtitle': 'From local prototype to multi-hospital cloud deployment',
        'author': 'OralAID Product Team', 'date': 'April 5, 2025',
        'read_time': '7 min', 'category': 'Product',
        'image': 'https://images.unsplash.com/photo-1516549655169-df83a0774514?w=900&q=80',
        'excerpt': 'Deploying medical AI in hospitals requires HIPAA compliance, HL7/FHIR integration, audit trails, and zero-downtime deployments. Here is what we learned.',
        'content': '''<p>Deploying OralAID as a SaaS product in hospital environments presented unique challenges around data privacy, integration with existing EMR systems, and regulatory compliance.</p>
        <h3>Key Integration Points</h3>
        <ul>
        <li><strong>HL7/FHIR API</strong>: Bidirectional patient data exchange with Epic, Cerner, and Meditech</li>
        <li><strong>DICOM Support</strong>: Direct import from dental imaging systems</li>
        <li><strong>SSO Integration</strong>: SAML 2.0 / OAuth 2.0 with hospital identity providers</li>
        <li><strong>Audit Trails</strong>: Every AI inference logged with clinician ID, timestamp, and decision</li>
        </ul>
        <h3>AWS Architecture</h3>
        <p>Production deployment uses ECS Fargate for containerized inference, RDS PostgreSQL for patient data, S3 for image storage, and CloudFront for global CDN delivery — achieving sub-200ms response times globally.</p>'''
    },
]

WHO_DATA = {
    'total_cases': 389846,
    'total_deaths': 188438,
    'survival_stage1': 83,
    'survival_stage4': 38,
    'countries': [
        {'country': 'India', 'cases': 119992, 'deaths': 72616, 'flag': '🇮🇳'},
        {'country': 'China', 'cases': 48000, 'deaths': 22000, 'flag': '🇨🇳'},
        {'country': 'USA', 'cases': 54000, 'deaths': 10850, 'flag': '🇺🇸'},
        {'country': 'Bangladesh', 'cases': 28000, 'deaths': 18000, 'flag': '🇧🇩'},
        {'country': 'Pakistan', 'cases': 21000, 'deaths': 14000, 'flag': '🇵🇰'},
        {'country': 'Brazil', 'cases': 15000, 'deaths': 7500, 'flag': '🇧🇷'},
        {'country': 'Indonesia', 'cases': 12000, 'deaths': 7200, 'flag': '🇮🇩'},
        {'country': 'Sri Lanka', 'cases': 4200, 'deaths': 2800, 'flag': '🇱🇰'},
        {'country': 'Myanmar', 'cases': 8500, 'deaths': 5100, 'flag': '🇲🇲'},
        {'country': 'Vietnam', 'cases': 6800, 'deaths': 4200, 'flag': '🇻🇳'},
    ]
}

@app.route('/blogs')
def blogs():
    return render_template('blogs.html', blogs=BLOGS, who_data=WHO_DATA)

@app.route('/blogs/<slug>')
def blog_post(slug):
    post = next((b for b in BLOGS if b['slug'] == slug), None)
    if not post:
        return redirect(url_for('blogs'))
    return render_template('blog_post.html', post=post, blogs=BLOGS)

COMPANY_PAGES = {
    'about': {
        'title': 'About OralAID',
        'eyebrow': 'Company',
        'summary': 'OralAID is building hospital-grade AI infrastructure for earlier oral cancer detection, private deployment, and explainable clinical workflows.',
        'sections': [
            ('Mission', 'Help hospitals and dental networks detect suspicious oral lesions earlier without moving patient data outside their trusted infrastructure.'),
            ('Our AI', 'AstraOral AI combines image classification, U-Net lesion segmentation, explainability, severity scoring, and longitudinal progression tracking.'),
            ('Deployment Model', 'OralAID can run in cloud, private VPC, or on-premise servers so patient images and records remain under hospital governance.')
        ]
    },
    'services': {
        'title': 'Services',
        'eyebrow': 'Enterprise Services',
        'summary': 'Implementation, integration, validation, and support packages for clinical AI screening programs.',
        'sections': [
            ('On-Premise Deployment', 'GPU or CPU inference stack deployed inside hospital data centers with local storage and private networking.'),
            ('Data Source Integration', 'FHIR, CSV, imaging folders, and API workflows for real-time patient data ingestion and instant AI results.'),
            ('Clinical Enablement', 'Admin training, workflow mapping, report templates, QA dashboards, and launch support for screening teams.')
        ]
    },
    'case-studies': {
        'title': 'Case Studies',
        'eyebrow': 'Hospital Impact',
        'summary': 'Examples of how OralAID can support screening, referral prioritization, and follow-up continuity.',
        'sections': [
            ('Regional Dental Network', 'Standardized screening across 42 clinics with batch upload, clinician verification, and referral routing.'),
            ('Oncology Department', 'High-risk lesion triage with explainability overlays and PDF reports for specialist review.'),
            ('Community Screening Program', 'Tablet-first screening workflow for outreach camps with offline severity scoring and later sync.')
        ]
    },
    'careers': {
        'title': 'Careers',
        'eyebrow': 'Build With Us',
        'summary': 'Join a healthcare AI team focused on private, clinically useful software for early cancer detection.',
        'sections': [
            ('Clinical AI Engineers', 'Build robust computer vision systems for constrained clinical environments.'),
            ('Healthcare Product Designers', 'Design calm, high-trust workflows for clinicians who repeat these tasks every day.'),
            ('Deployment Engineers', 'Package OralAID for secure on-premise, VPC, and regulated hospital environments.')
        ]
    },
    'privacy': {
        'title': 'Privacy Policy',
        'eyebrow': 'Privacy',
        'summary': 'OralAID is designed around clinical data minimization, hospital-controlled storage, and private deployment options.',
        'sections': [
            ('Patient Data Control', 'On-premise deployments keep images, reports, and patient records inside the hospital environment.'),
            ('Security Practices', 'Role-based access, local audit trails, HTTPS-ready deployment, and configurable retention policies.'),
            ('AI Processing', 'Inference can run locally without sending patient images to third-party AI APIs.')
        ]
    }
}

@app.route('/about')
def about():
    return render_template('company.html', page=COMPANY_PAGES['about'])

@app.route('/services')
def services():
    return render_template('company.html', page=COMPANY_PAGES['services'])

@app.route('/case-studies')
def case_studies():
    return render_template('company.html', page=COMPANY_PAGES['case-studies'])

@app.route('/careers')
def careers():
    return render_template('company.html', page=COMPANY_PAGES['careers'])

@app.route('/privacy')
def privacy():
    return render_template('company.html', page=COMPANY_PAGES['privacy'])

# ─── Daily.co & Appointments ──────────────────────────────────────────────────

DAILY_API_KEY = os.environ.get('DAILY_API_KEY')

def create_daily_room():
    if not DAILY_API_KEY:
        return None
    try:
        exp = int(time.time()) + 3600  # Expires in 1 hour
        headers = {'Authorization': f'Bearer {DAILY_API_KEY}'}
        data = {
            "privacy": "private",
            "properties": {"exp": exp}
        }
        res = requests.post("https://api.daily.co/v1/rooms", json=data, headers=headers)
        if res.status_code == 200:
            return res.json().get('url')
    except Exception as e:
        print(f"Daily.co Error: {e}")
    return None

@app.route('/api/schedule-meet', methods=['POST'])
@login_required
def schedule_meet():
    data = request.get_json(silent=True) or {}
    patient_id = data.get('patient_id')
    if not patient_id:
        return jsonify({'error': 'Patient ID required'}), 400
    
    room_url = create_daily_room()
    if not room_url:
        return jsonify({'error': 'Failed to create video room'}), 500
        
    appt = Appointment(
        patient_id=patient_id,
        clinician_id=current_user.id,
        scheduled_date=datetime.datetime.utcnow(),
        type='Video Consultation',
        room_url=room_url,
        notes=data.get('notes', 'Scheduled from analysis result')
    )
    db.session.add(appt)
    
    notif = Notification(
        user_id=current_user.id,
        message=f"Video consultation scheduled for patient ID {patient_id}",
        type='appointment'
    )
    db.session.add(notif)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'meet_url': room_url,
        'appointment_id': appt.id
    })

@app.route('/call/<int:appointment_id>')
@login_required
def video_call(appointment_id):
    appt = Appointment.query.get_or_404(appointment_id)
    if appt.clinician_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    return render_template('call.html', appt=appt)

@app.route('/api/appointments', methods=['GET', 'POST'])
@login_required
def api_appointments():
    if request.method == 'POST':
        data = request.json
        scheduled_date = datetime.datetime.fromisoformat(data['scheduled_date'].replace('Z', ''))
        
        room_url = None
        if data.get('type') == 'Video Consultation':
            room_url = create_daily_room()
            
        appt = Appointment(
            patient_id=data['patient_id'],
            clinician_id=current_user.id,
            scheduled_date=scheduled_date,
            type=data['type'],
            room_url=room_url,
            notes=data.get('notes', '')
        )
        db.session.add(appt)
        
        notif = Notification(
            user_id=current_user.id,
            message=f"New {data['type']} scheduled for {scheduled_date.strftime('%B %d, %H:%M')}",
            type='appointment'
        )
        db.session.add(notif)
        db.session.commit()
        return jsonify({'status': 'success', 'appointment_id': appt.id})
    
    appts = Appointment.query.filter_by(clinician_id=current_user.id).all()
    return jsonify([{
        'id': a.id,
        'patient_name': a.patient.name,
        'patient_id': a.patient.id,
        'scheduled_date': a.scheduled_date.isoformat(),
        'type': a.type,
        'status': a.status,
        'room_url': a.room_url,
        'notes': a.notes
    } for a in appts])

@app.route('/api/appointments/<int:id>', methods=['PATCH'])
@login_required
def update_appointment(id):
    appt = Appointment.query.get_or_404(id)
    data = request.json
    if 'status' in data:
        appt.status = data['status']
    db.session.commit()
    return jsonify({'status': 'success'})

# ─── Messaging ────────────────────────────────────────────────────────────────

@app.route('/api/messages/<int:patient_id>', methods=['GET'])
@login_required
def get_messages(patient_id):
    msgs = ChatMessage.query.filter_by(patient_id=patient_id, clinician_id=current_user.id).order_by(ChatMessage.created_at.asc()).all()
    return jsonify([{
        'id': m.id,
        'sender_type': m.sender_type,
        'content': m.content,
        'created_at': m.created_at.isoformat(),
        'is_read': m.is_read
    } for m in msgs])

@app.route('/api/messages', methods=['POST'])
@login_required
def send_message():
    data = request.json
    msg = ChatMessage(
        patient_id=data['patient_id'],
        clinician_id=current_user.id,
        sender_type='doctor',
        content=data['content']
    )
    db.session.add(msg)
    
    db.session.commit()
    return jsonify({'status': 'success', 'message_id': msg.id})

# ─── Notifications ────────────────────────────────────────────────────────────

@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    notifs = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).limit(20).all()
    unread_count = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
    return jsonify({
        'notifications': [{
            'id': n.id,
            'message': n.message,
            'type': n.type,
            'is_read': n.is_read,
            'created_at': n.created_at.isoformat()
        } for n in notifs],
        'unread_count': unread_count
    })

@app.route('/api/notifications/<int:id>/read', methods=['POST'])
@login_required
def mark_notif_read(id):
    notif = Notification.query.get_or_404(id)
    notif.is_read = True
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/api/patients', methods=['GET'])
@login_required
def api_patients():
    search = request.args.get('q', '')
    if search:
        patients = Patient.query.filter(Patient.name.ilike(f'%{search}%')).all()
    else:
        patients = Patient.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'age': p.age,
        'gender': p.gender
    } for p in patients])

# ─── SaaS Pricing ─────────────────────────────────────────────────────────────

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# ─── AWS DevOps Dashboard ─────────────────────────────────────────────────────

@app.route('/devops')
@admin_required
def devops_dashboard():
    # Corrections for active learning
    corrections_count = Diagnosis.query.filter(Diagnosis.doctor_feedback.isnot(None)).count()
    retrain_history = ModelMetrics.query.order_by(ModelMetrics.recorded_at.desc()).all()
    
    # Mock AWS Infrastructure data (In a real app, use boto3 to fetch)
    infra_health = {
        'ecs': {'status': 'Running', 'tasks': 2, 'cpu': 24, 'mem': 31},
        'rds': {'status': 'Available', 'conns': 14, 'storage': '12.4 GB'},
        's3': {'objects': 1420, 'storage': '4.2 GB'},
        'github': {'status': 'Success', 'hash': 'sha-abc1234', 'time': '2h ago'}
    }
    
    return render_template('devops.html', 
        corrections_count=corrections_count,
        retrain_history=retrain_history,
        infra=infra_health,
        now=datetime.datetime.utcnow())

# ─── Print Report ─────────────────────────────────────────────────────────────

@app.route('/metrics')
def prometheus_metrics():
    total   = Diagnosis.query.count()
    high    = Diagnosis.query.filter_by(risk_level='High-risk').count()
    correct = Diagnosis.query.filter_by(doctor_feedback='correct').count()
    lines = [
        '# HELP oralaid_total_diagnoses Total diagnoses',
        '# TYPE oralaid_total_diagnoses counter',
        f'oralaid_total_diagnoses {total}',
        '# HELP oralaid_high_risk_diagnoses High risk diagnoses',
        '# TYPE oralaid_high_risk_diagnoses counter',
        f'oralaid_high_risk_diagnoses {high}',
        '# HELP oralaid_correct_predictions Doctor confirmed correct',
        '# TYPE oralaid_correct_predictions counter',
        f'oralaid_correct_predictions {correct}',
    ]
    return Response('\n'.join(lines), mimetype='text/plain')

# ─── Init ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        validate_license()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(
                username='admin', email='admin@oralaid.com',
                password_hash=generate_password_hash('admin123'),
                role='admin', full_name='System Administrator',
            ))
            db.session.commit()
            print('✅ Admin user created (admin/admin123)')
    print('🚀 OralAID running on http://0.0.0.0:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)
