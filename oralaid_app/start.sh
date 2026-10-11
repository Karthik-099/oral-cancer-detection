#!/bin/bash

echo "🦷 Starting OralAID Application..."
echo "=================================="
echo ""

cd /home/karthik/oral-cancer-detection/oralaid_app

# Check if database exists, if not create it
if [ ! -f "oralaid.db" ]; then
    echo "📊 Creating database..."
    python3 -c "
from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@oralaid.com',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            full_name='System Administrator'
        )
        db.session.add(admin)
        db.session.commit()
        print('✅ Admin user created')
"
fi

echo ""
echo "✅ OralAID is ready!"
echo ""
echo "🌐 Access the application at: http://localhost:5000"
echo "👤 Login credentials: admin / admin123"
echo ""
echo "📋 Features:"
echo "   ✓ AI-powered oral cancer detection"
echo "   ✓ LIME & GradCAM visualizations"
echo "   ✓ Risk assessment & recommendations"
echo "   ✓ Patient management"
echo "   ✓ Clinical reporting"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

python3 app.py