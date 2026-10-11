from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    return jsonify({'result': 'Normal', 'confidence': 0.95})

if __name__ == '__main__':
    print('Server running at http://localhost:5001')
    app.run(debug=True, host='0.0.0.0', port=5001)
