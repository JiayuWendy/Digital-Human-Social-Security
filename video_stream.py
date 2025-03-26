"""
video_stream.py
摄像头实时视频流：
- 注册：采集多角度人脸图像
- 实时认证：持续检测用户并返回检测结果
"""
import cv2
import os
import time
from datetime import datetime
import face_recognition
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ✅ 自动检测可用摄像头
def auto_detect_camera():
    """自动检测可用摄像头索引"""
    for i in range(5):  # 检测索引 0-4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ 摄像头索引可用: {i}")
            cap.release()
            return i
        cap.release()
    print("❌ 没有可用的摄像头")
    return -1


# ✅ 中文显示函数
def put_chinese_text(frame, text, position, font_path="msyh.ttc", font_size=24, color=(0, 255, 0)):
    """在 OpenCV 窗口上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"❌ 未找到字体文件：{font_path}")
        return frame

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ✅ 修改摄像头认证函数（保持摄像头开启）
def recognize_camera(encoding_dir="./encodings", timeout=120, keep_open_duration=30, font_path="msyh.ttc"):
    """
    实时认证：
    - 从摄像头检测人脸
    - 成功后摄像头保持开启一段时间
    参数：
    - encoding_dir: 特征编码文件夹路径
    - timeout: 最大检测时间（秒）
    - keep_open_duration: 检测成功后摄像头保持开启的时间（秒）
    - font_path: 中文字体路径
    """

    # 自动选择可用摄像头
    camera_index = auto_detect_camera()
    if camera_index == -1:
        print("❌ 无法找到可用摄像头")
        return None

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return None

    start_time = time.time()
    encodings = load_encodings(encoding_dir)

    if not encodings:
        print("❌ 无人脸编码文件")
        cap.release()
        cv2.destroyAllWindows()
        return None

    authenticated = False
    auth_start_time = None
    recognized_user = None
    confidence = 0.0

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频流")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_encodings:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                for name, encoding in encodings.items():
                    results = face_recognition.compare_faces([encoding], face_encoding)
                    face_distance = face_recognition.face_distance([encoding], face_encoding)[0]

                    if results[0]:
                        confidence = 1 - face_distance
                        msg = f"识别成功: {name} ({confidence:.2f})"
                        frame = put_chinese_text(frame, msg, (20, 50), font_path=font_path)

                        display_face(frame, face_location, name, confidence)

                        if not authenticated:
                            authenticated = True
                            auth_start_time = time.time()
                            recognized_user = name

        # 显示认证中或认证成功
        if authenticated:
            elapsed_since_auth = time.time() - auth_start_time
            if elapsed_since_auth > keep_open_duration:
                print(f"✅ 摄像头已保持开启 {keep_open_duration} 秒，关闭摄像头")
                break
            msg = f"✅ 已认证: {recognized_user} ({confidence:.2f})"
        else:
            msg = "⏳ 检测中..."

        frame = put_chinese_text(frame, msg, (20, 50), font_path=font_path)
        cv2.imshow("实时认证", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if authenticated:
        print(f"✅ 认证成功: {recognized_user} ({confidence:.2f})")
        return recognized_user, confidence
    else:
        print("⏱️ 超时，认证失败")
        return None


# ✅ 加载人脸编码
def load_encodings(encoding_dir):
    """加载人脸编码"""
    encodings = {}

    if not os.path.exists(encoding_dir):
        print(f"❌ 编码目录不存在：{encoding_dir}")
        return encodings

    for filename in os.listdir(encoding_dir):
        if filename.endswith(".pkl"):
            name = filename.split(".")[0]
            with open(os.path.join(encoding_dir, filename), "rb") as f:
                encodings[name] = pickle.load(f)

    print(f"✅ 加载 {len(encodings)} 个编码")
    return encodings


# ✅ 显示人脸与标签
def display_face(frame, face_location, name, confidence):
    """在视频流中显示人脸框和标签"""
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{name} ({confidence:.2f})"
    frame = put_chinese_text(frame, label, (left, top - 30), font_path="msyh.ttc")
    cv2.imshow("实时认证", frame)
