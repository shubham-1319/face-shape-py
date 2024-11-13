from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Load Dlib's pre-trained shape predictor
predictor_path = r"C:/Users/Shubh/Downloads/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

@app.route('/detect_face_shape', methods=['POST'])
def detect_face_shape():
    # Get the image file from the request
    image_file = request.files['image']
    
    # Read the image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Unable to load image"}), 400

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

        # Calculate measurements for face shape
        jaw_width = np.linalg.norm(points[0] - points[16])
        forehead_width = np.linalg.norm(points[17] - points[26])
        face_length = np.linalg.norm(points[8] - points[27])

        # Determine face shape based on measurements
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

if __name__ == '__main__':
    app.run(debug=True)
