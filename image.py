from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Path to the shape predictor file
predictor_path = os.getenv("PREDICTOR_PATH", "shape_predictor_68_face_landmarks.dat")

# Ensure the shape predictor file exists
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Predictor file not found at {predictor_path}")

# Load Dlib's pre-trained shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

@app.route('/detect_face_shape', methods=['POST'])
def detect_face_shape():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Unable to load image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return jsonify({
            "error": "No face detected. Please try again or go outside for better lighting."
        }), 400

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

        jaw_width = np.linalg.norm(points[0] - points[16])
        forehead_width = np.linalg.norm(points[17] - points[26])
        face_length = np.linalg.norm(points[8] - points[27])

        if forehead_width > jaw_width and face_length > jaw_width:
            face_shape = "Oval"
        elif jaw_width > forehead_width and face_length > forehead_width:
            face_shape = "Square"
        elif jaw_width > face_length:
            face_shape = "Round"
        elif forehead_width > jaw_width:
            face_shape = "Heart"
        else:
            face_shape = "Diamond"

        return jsonify({"face_shape": face_shape})

    return jsonify({"error": "Face shape could not be determined"}), 400

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
