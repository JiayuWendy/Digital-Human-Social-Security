"""
face_recog.py
人脸识别模块：
- 注册人脸
- 实时认证
- 模型训练与保存
"""
import os
import face_recognition
import numpy as np
import cv2
import pickle


class FaceAuth:
    def __init__(self, encoding_dir="./encodings"):
        """初始化编码目录"""
        self.encoding_dir = encoding_dir
        os.makedirs(self.encoding_dir, exist_ok=True)

    def save_encoding(self, name, encoding):
        """保存人脸编码到文件"""
        encoding_path = os.path.join(self.encoding_dir, f"{name}.pkl")
        with open(encoding_path, "wb") as f:
            pickle.dump(encoding, f)
        print(f"✅ 模型保存成功：{encoding_path}")

    def load_encoding(self, name):
        """加载指定用户的编码"""
        encoding_path = os.path.join(self.encoding_dir, f"{name}.pkl")
        if not os.path.exists(encoding_path):
            print(f"❌ 未找到编码文件：{encoding_path}")
            return None

        with open(encoding_path, "rb") as f:
            encoding = pickle.load(f)
        return encoding

    def train_model_from_folder(self, img_dir, name):
        """
        从整个文件夹训练模型：
        - 遍历文件夹中的所有图片
        - 编码并计算平均向量
        """
        encodings = []

        # 遍历文件夹中的所有图片
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)

            # 确保只处理图片
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    encodings.append(face_encodings[0])
                else:
                    print(f"❌ 未检测到人脸: {img_path}")

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            self.save_encoding(name, avg_encoding)
            print(f"✅ 模型训练完成: {name}")
            return True
        else:
            print(f"❌ 模型训练失败：{name}")
            return False

    def recognize_face(self, frame):
        """检测帧中的人脸并返回用户与置信度"""
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings) == 0:
            return None

        detected_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            for filename in os.listdir(self.encoding_dir):
                if filename.endswith(".pkl"):
                    name = filename.split(".")[0]
                    known_encoding = self.load_encoding(name)

                    if known_encoding is not None:
                        results = face_recognition.compare_faces([known_encoding], face_encoding)
                        face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

                        if results[0]:
                            detected_faces.append((name, 1 - face_distance, face_location))

        return detected_faces if detected_faces else None

    def display_face(self, frame, face_location, name, confidence):
        """在视频流中显示人脸框和标签"""
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
