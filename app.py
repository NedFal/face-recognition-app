from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'reference' not in request.files or 'to_check' not in request.files:
        return jsonify({"message": "Images not provided"}), 400

    ref_file = request.files['reference']
    check_file = request.files['to_check']

    ref_image = face_recognition.load_image_file(io.BytesIO(ref_file.read()))
    check_image = face_recognition.load_image_file(io.BytesIO(check_file.read()))

    ref_encodings = face_recognition.face_encodings(ref_image)
    check_encodings = face_recognition.face_encodings(check_image)

    if not ref_encodings or not check_encodings:
        return jsonify({"message": "No face detected in one or both images"}), 400

    result = face_recognition.compare_faces([ref_encodings[0]], check_encodings[0], tolerance=0.6)
    return jsonify({"result": "Same person" if result[0] else "Different persons"})

if __name__ == '__main__':
    app.run()
