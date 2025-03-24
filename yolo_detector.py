"""
YOLOv8 人脸检测

将摄像头图像或上传图片传入 YOLOv8

YOLOv8 输出人脸边界框（x, y, w, h）

将检测到的人脸区域裁剪并送入 face_recognition 进行比对

"""
from ultralytics import YOLO
import cv2

class YOLOv8Detector:
    def __init__(self, model_path="./models/yolov8n.pt"):
        """
        初始化 YOLOv8 模型
        """
        self.model = YOLO(model_path)

    def detect_faces(self, frame, confidence_threshold=0.5):
        """
        YOLOv8 人脸检测
        :param frame: 视频帧
        :param confidence_threshold: 置信度阈值
        :return: 检测到的人脸边界框列表
        """
        results = self.model.predict(frame, conf=confidence_threshold)
        faces = []

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()

                # 筛选置信度高于阈值的框
                if conf > confidence_threshold:
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    faces.append((x1, y1, x2, y2, face))

        return faces
