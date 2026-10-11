#!/usr/bin/env python3
"""
Database initialization script for OralAID
Run this to set up the database with proper schema
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, User, Patient, Diagnosis, Annotation, ChatGroup, GroupMember, ChatMessage, Appointment, Document, LabContact
from werkzeug.security import generate_password_hash
import datetime

def init_database():
    """Initialize database with proper schema and sample data"""
    
    print("🔄 Initializing OralAID database...")
    
    with app.app_context():
        # Drop all tables and recreate
        db.drop_all()
        db.create_all()
        
        print("✅ Database schema created")
        
        # Create default users
        admin = User(
            username='admin',
            email='admin@oralaid.com',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            full_name='System Administrator',
            is_online=False
        )
        db.session.add(admin)
        
        doctor = User(
            username='doctor',
            email='doctor@oralaid.com',
            password_hash=generate_password_hash('doctor123'),
            role='doctor',
            full_name='Dr. John Smith',
            specialization='Oral Medicine',
            license_number='MD12345',
            hospital='General Hospital',
            phone='(555) 123-4567',
            is_online=False
        )
        db.session.add(doctor)
        
        patient_user = User(
            username='patient',
            email='patient@oralaid.com',
            password_hash=generate_password_hash('patient123'),
            role='patient',
            full_name='Jane Doe',
            phone='(555) 987-6543',
            is_online=False
        )
        db.session.add(patient_user)
        
        # Create additional demo doctors
        demo_doctors = [
            {
                'username': 'dr_wilson',
                'email': 'wilson@oralaid.com',
                'full_name': 'Dr. Sarah Wilson',
                'specialization': 'Pathology',
                'license_number': 'MD67890',
                'hospital': 'Medical Center'
            },
            {
                'username': 'dr_chen',
                'email': 'chen@oralaid.com',
                'full_name': 'Dr. Michael Chen',
                'specialization': 'Radiology',
                'license_number': 'MD11111',
                'hospital': 'Imaging Center'
            },
            {
                'username': 'dr_brown',
                'email': 'brown@oralaid.com',
                'full_name': 'Dr. Emily Brown',
                'specialization': 'Oncology',
                'license_number': 'MD22222',
                'hospital': 'Cancer Center'
            }
        ]
        
        for doc_data in demo_doctors:
            demo_doc = User(
                username=doc_data['username'],
                email=doc_data['email'],
                password_hash=generate_password_hash('demo123'),
                role='doctor',
                full_name=doc_data['full_name'],
                specialization=doc_data['specialization'],
                license_number=doc_data['license_number'],
                hospital=doc_data['hospital'],
                phone='(555) 000-0000',
                is_online=False
            )
            db.session.add(demo_doc)
        
        # Create sample lab contacts
        labs = [
            LabContact(
                name='Advanced Pathology Services',
                lab_type='pathology',
                contact_person='Dr. Sarah Wilson',
                phone='(555) 123-4567',
                email='results@advancedpath.com',
                address='123 Medical Center Dr, Suite 200',
                services='Oral Pathology, Histopathology, Immunohistochemistry',
                turnaround_time='3-5 business days',
                is_active=True
            ),
            LabContact(
                name='Metro Imaging Center',
                lab_type='radiology',
                contact_person='Dr. Michael Chen',
                phone='(555) 987-6543',
                email='scheduling@metroimaging.com',
                address='456 Healthcare Blvd, Floor 3',
                services='CT, MRI, X-Ray, Ultrasound',
                turnaround_time='Same day',
                is_active=True
            ),
            LabContact(
                name='QuickTest Laboratory',
                lab_type='blood_work',
                contact_person='Lab Manager',
                phone='(555) 555-0123',
                email='results@quicktest.com',
                address='789 Lab Street, Building A',
                services='CBC, Metabolic Panel, Tumor Markers',
                turnaround_time='24-48 hours',
                is_active=True
            ),
            LabContact(
                name='Precision Microbiology',
                lab_type='microbiology',
                contact_person='Dr. Lisa Johnson',
                phone='(555) 444-5555',
                email='micro@precisionlab.com',
                address='321 Science Park Dr',
                services='Bacterial Culture, Sensitivity Testing, Viral PCR',
                turnaround_time='2-7 days',
                is_active=True
            )
        ]
        
        for lab in labs:
            db.session.add(lab)
        
        # Create sample chat groups
        groups = [
            ChatGroup(
                name='Oral Medicine',
                description='Discussion group for oral medicine specialists',
                group_type='medical',
                created_by=2  # doctor user
            ),
            ChatGroup(
                name='Pathology',
                description='Pathology consultation and case discussions',
                group_type='medical',
                created_by=2
            ),
            ChatGroup(
                name='Oncology',
                description='Oral cancer treatment and management',
                group_type='medical',
                created_by=2
            ),
            ChatGroup(
                name='Radiology',
                description='Imaging and diagnostic radiology',
                group_type='medical',
                created_by=2
            )
        ]
        
        for group in groups:
            db.session.add(group)
        
        # Commit all changes
        db.session.commit()
        
        print("✅ Sample data created")
        print("\n🎯 Default User Accounts:")
        print("   Admin:   admin / admin123")
        print("   Doctor:  doctor / doctor123")
        print("   Patient: patient / patient123")
        print("\n🏥 Demo Doctor Accounts:")
        print("   Dr. Wilson: dr_wilson / demo123")
        print("   Dr. Chen:   dr_chen / demo123")
        print("   Dr. Brown:  dr_brown / demo123")
        print("\n🚀 Database initialization complete!")

if __name__ == '__main__':
    init_database()