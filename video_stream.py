"""
实时视频流检测

功能演示
打开摄像头，实时检测人脸
显示人脸框、姓名与置信度
按下 q 键退出
"""

import cv2
import face_recognition
import numpy as np
from yolo_detector import YOLOv8Detector
from face_recognition import FaceAuth

# 初始化 YOLOv8 检测器和人脸识别模块
MODEL_PATH = "./models/yolov8n.pt"
detector = YOLOv8Detector(MODEL_PATH)
auth = FaceAuth()

def draw_results(frame, name, confidence, x1, y1, x2, y2):
    """
    在视频流上绘制识别结果

    -在视频流上绘制人脸框与名称
    -置信度 > 60% 显示绿色，低于 60% 显示红色
    """
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

    # 绘制人脸框
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 显示名称和置信度
    label = f"{name} ({confidence}%)" if confidence else "Unknown"
    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_frame(frame):
    """
    处理每一帧视频流：
    - YOLOv8 检测人脸
    - 人脸识别比对
    - 显示识别结果
    """
    faces = detector.detect_faces(frame)

    for x1, y1, x2, y2, face in faces:
        if face is not None and face.size > 0:
            # 人脸编码比对
            encodings = face_recognition.face_encodings(face)

            if encodings:
                encoding = encodings[0]
                distances = face_recognition.face_distance(auth.known_face_encodings, encoding)

                # 匹配最接近的人脸
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.6:
                    name = auth.known_face_names[best_match_index]
                    confidence = round((1 - distances[best_match_index]) * 100, 2)
                else:
                    name, confidence = "Unknown", 0

                # 绘制结果
                draw_results(frame, name, confidence, x1, y1, x2, y2)

    return frame

def start_video_stream():
    """
    启动摄像头视频流
    每帧调用 process_frame() 进行人脸检测与识别
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: 无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: 无法读取视频流！")
            break

        # 实时处理视频流
        frame = process_frame(frame)

        # 显示视频流
        cv2.imshow("YOLOv8 实时人脸识别", frame)

        # 按下 `q` 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_stream()
