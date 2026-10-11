import os
from app import app, db

# Delete old database
db_path = 'instance/oralaid.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✅ Deleted old database at {db_path}")
elif os.path.exists('oralaid.db'):
    os.remove('oralaid.db')
    print("✅ Deleted old database at root")

# Create new database
with app.app_context():
    db.create_all()
    print("✅ Created new database with correct schema")
    
    # Verify tables
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    print(f"✅ Tables created: {tables}")
    
    # Verify diagnosis columns
    if 'diagnosis' in tables:
        columns = [c['name'] for c in inspector.get_columns('diagnosis')]
        print(f"✅ Diagnosis columns: {columns}")
    
    # Create admin user
    from werkzeug.security import generate_password_hash
    from app import User
    
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
        print("✅ Created admin user (admin/admin123)")

print("\n✅ Database rebuild complete!")
