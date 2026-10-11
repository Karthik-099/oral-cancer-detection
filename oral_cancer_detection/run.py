#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'frontend'))
from app import app

if __name__ == '__main__':
    print('='*50)
    print('Oral Cancer Detection System')
    print('='*50)
    print('Server starting at http://localhost:5001')
    print('Press Ctrl+C to stop')
    print('='*50)
    app.run(debug=True, host='0.0.0.0', port=5001)
