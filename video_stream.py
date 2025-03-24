"""
video_stream.py

实时视频流检测

功能演示
打开摄像头，实时检测人脸
显示人脸框、姓名与置信度
按下 q 键退出
"""

"""
video_stream.py
摄像头实时视频流：
 - 实时采集多角度人脸
 - 自动保存至指定路径
"""
import cv2
import os
import time

def capture_face(save_dir, username):
    """
    使用摄像头实时采集人脸：
    - 正脸、左脸、右脸采集
    - 自动保存图片
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return None

    count = 0
    os.makedirs(save_dir, exist_ok=True)

    while count < 3:
        ret, frame = cap.read()

        if not ret:
            print("❌ 无法读取视频流")
            break

        msg = ["请正对摄像头", "请向左转脸", "请向右转脸"][count]
        cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("采集人脸", frame)

        img_path = os.path.join(save_dir, f"{username}_{count + 1}.jpg")
        cv2.imwrite(img_path, frame)

        print(f"✅ 已保存：{img_path}")
        time.sleep(2)

        count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ 采集完成，图片保存至 {save_dir}")
    return save_dir
