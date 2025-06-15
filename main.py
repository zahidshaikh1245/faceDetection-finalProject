import cv2
import mediapipe as mp
import numpy as np
from pymongo import MongoClient
import os
import json

# ===========================
# MongoDB Setup
# ===========================
client = MongoClient("mongodb+srv://zahidshaikh:zahidshaikh@cluster0.a5fck.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["faces"]

# Folder to save face images locally for training
if not os.path.exists("faces"):
    os.makedirs("faces")

# ===========================
# MediaPipe Setup
# ===========================
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# ===========================
# OpenCV LBPH Face Recognizer
# ===========================
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels():
    image_paths = [os.path.join("faces", f) for f in os.listdir("faces")]
    face_samples = []
    ids = []
    name_id_map = {}

    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.basename(image_path).split('.')[0])
        name = os.path.basename(image_path).split('.')[1]
        face_samples.append(gray_img)
        ids.append(id)
        name_id_map[id] = name
    return face_samples, ids, name_id_map

def train_recognizer():
    faces, ids, name_map = get_images_and_labels()
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        return name_map
    else:
        return {}

# ===========================
# Camera Setup
# ===========================
cap = cv2.VideoCapture(0)

# Load initial trained data
name_id_map = train_recognizer()
next_id = max(name_id_map.keys()) + 1 if name_id_map else 0

# ===========================
# Main Loop
# ===========================
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.6) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                x, y = max(0, x), max(0, y)
                face_crop = gray[y:y+h, x:x+w]

                label_text = "Unknown"

                if face_crop.size > 0 and len(name_id_map) > 0:
                    id_, confidence = recognizer.predict(face_crop)
                    if confidence < 60:
                        label_text = name_id_map[id_]
                    else:
                        label_text = "Unknown"
                else:
                    label_text = "Unknown"

                if label_text == "Unknown":
                    cv2.imshow("Unknown Face", face_crop)
                    cv2.waitKey(1)
                    new_name = input("New face detected! Enter name: ")
                    img_name = f"{next_id}.{new_name}.jpg"
                    cv2.imwrite(f"faces/{img_name}", face_crop)

                    # Get face mesh landmarks
                    face_roi_rgb = rgb[y:y+h, x:x+w]
                    mesh_results = face_mesh.process(face_roi_rgb)

                    face_mesh_points = []
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            for lm in face_landmarks.landmark:
                                # Normalize to ROI size
                                px = int(lm.x * w)
                                py = int(lm.y * h)
                                pz = lm.z
                                face_mesh_points.append({"x": px, "y": py, "z": pz})

                    # Insert info to DB
                    collection.insert_one({
                        "id": next_id,
                        "name": new_name,
                        "face_mesh": face_mesh_points
                    })

                    next_id += 1
                    name_id_map = train_recognizer()
                    cv2.destroyWindow("Unknown Face")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Face Detection & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
