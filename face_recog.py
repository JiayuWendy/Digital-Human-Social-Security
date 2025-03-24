"""
face_recog.py

人脸识别逻辑
"""

import os
import pickle

import numpy as np

import face_recognition

from yolo_detector import YOLOv8Detector


class FaceAuth:
    def __init__(self, db_path='./data/face_db.pkl', model_path='./models/yolov8n.pt'):
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.detector = YOLOv8Detector(model_path)

        # 加载已存储的数据库
        if os.path.exists(db_path):
            with open(db_path, 'rb') as db_file:
                data = pickle.load(db_file)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']

    # 注册人脸
    def register_face(self, name, img_path):
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)

        if encodings:
            encoding = encodings[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

            # 保存数据库
            self.save_db()
            return True
        return False

    # 保存数据库
    def save_db(self):
        with open(self.db_path, 'wb') as db_file:
            pickle.dump({
                "encodings": self.known_face_encodings,
                "names": self.known_face_names
            }, db_file)

    # 验证人脸
    def verify_face(self, img_path):
        faces = self.detector.detect_faces(img_path)

        if faces:
            for face in faces:
                encoding = face_recognition.face_encodings(face)

                if encoding:
                    unknown_encoding = encoding[0]
                    distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)

                    best_match_index = np.argmin(distances)

                    if distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                        confidence = round((1 - distances[best_match_index]) * 100, 2)
                        return name, confidence
        return None, 0
