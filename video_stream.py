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


# ✅ 修改摄像头采集函数
def capture_face_motion(save_dir, username, duration=15, fps=5, font_path="msyh.ttc"):
    """使用摄像头实时采集人脸多帧"""

    camera_index = auto_detect_camera()  # 自动选择可用摄像头
    if camera_index == -1:
        print("❌ 无法找到可用摄像头")
        return []

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return []

    os.makedirs(save_dir, exist_ok=True)

    print("\n📸 请缓慢左右晃动头部进行采集...")
    print(f"⏱️ 采集时间：{duration} 秒，每秒采集 {fps} 帧")

    img_paths = []
    frame_interval = 1 / fps
    start_time = time.time()
    last_capture_time = start_time
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频流")
            break

        elapsed_time = time.time() - start_time
        progress = (elapsed_time / duration) * 100
        msg = f"请左右缓慢晃头采集人脸信息: {progress:.1f}%"

        frame = put_chinese_text(frame, msg, (20, 50), font_path=font_path)

        cv2.imshow("人脸采集", frame)

        if time.time() - last_capture_time >= frame_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(save_dir, f"{username}_{timestamp}_{frame_count}.jpg")
            cv2.imwrite(img_path, frame)
            img_paths.append(img_path)
            print(f"✅ 已保存: {img_path}")

            frame_count += 1
            last_capture_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ 采集完成！共采集 {len(img_paths)} 张图片")
    return img_paths


# ✅ 修改摄像头认证函数（支持中文标签）
def recognize_camera(encoding_dir="./encodings", timeout=60, detection_duration=20, font_path="msyh.ttc"):
    """实时认证：从摄像头检测人脸，并显示中文标签"""

    camera_index = auto_detect_camera()  # 自动选择可用摄像头
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

    # 设定每次摄像头开启的时间段
    camera_start_time = time.time()
    best_match = None  # 最佳匹配人脸
    best_confidence = 0  # 最高置信度
    best_name = None  # 最佳匹配的名字

    # 在 timeout 时间内一直进行检测
    while time.time() - start_time < timeout:
        # 检测时间段
        detection_start_time = time.time()

        while time.time() - detection_start_time < detection_duration:
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
                            # 比较当前检测到的置信度是否为最高
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_name = name
                                best_match = face_location

                            msg = f"识别中: {name} ({confidence:.2f})"
                            frame = put_chinese_text(frame, msg, (20, 50), font_path=font_path)

            # 显示检测中的信息
            if best_match is not None:
                display_face(frame, best_match, best_name, best_confidence, font_path)

            # 在没有检测到最佳人脸时，显示“检测中”
            if best_match is None:
                frame = put_chinese_text(frame, "⏳ 检测中...", (20, 50), font_path=font_path)

            cv2.imshow("实时认证", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if best_match is not None and best_confidence > 0:
            print(f"✅ 检测结果: {best_name} ({best_confidence:.2f})")
            cap.release()
            cv2.destroyAllWindows()
            return best_name, best_confidence

        if time.time() - camera_start_time > timeout:
            print(f"⏱️ 超过最大检测时间: {timeout}秒，认证失败")
            cap.release()
            cv2.destroyAllWindows()
            break

    print("❌ 未检测到有效的人脸，认证失败")
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


# ✅ 显示人脸与标签（支持中文标签）
def display_face(frame, face_location, name, confidence, font_path="msyh.ttc"):
    """在视频流中显示人脸框和标签"""
    top, right, bottom, left = face_location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    label = f"{name} ({confidence:.2f})"
    frame = put_chinese_text(frame, label, (left, top - 30), font_path=font_path)
    cv2.imshow("实时认证", frame)