"""
main.py

FastAPI服务
人脸识别模块

功能说明：

FastAPI服务：
 - 人脸注册与验证
 - 支持实时摄像头采集
 - 将人脸特征存储在数据库中

使用说明：
    运行 FastAPI 服务：
    -uvicorn main:app --reload

    使用 Postman 测试：
    -/register 注册人脸：
    POST http://localhost:8000/register?name=Alice 上传图片：`face1.jpg`
    -/verify 验证人脸
    POST http://localhost:8000/verify 上传图片：`face2.jpg`
"""
"""
main.py
FastAPI服务：
 - 人脸注册与验证
 - 支持实时摄像头采集
 - 将人脸特征存储在数据库中
"""

import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from face_recog import FaceAuth
from db import init_db, save_user, load_users
from video_stream import capture_face

app = FastAPI()

# 初始化数据库
init_db()
face_auth = FaceAuth()


# 注册人脸接口
@app.post("/register")
async def register(name: str, file: UploadFile = File(None)):
    """
    注册人脸：
    - 上传图片或通过摄像头采集
    - 将人脸存储在数据库中
    """
    user_dir = f"./data/{name}"
    os.makedirs(user_dir, exist_ok=True)

    if file:
        # 文件上传模式
        img_path = f"{user_dir}/{name}.jpg"
        with open(img_path, "wb") as img_file:
            shutil.copyfileobj(file.file, img_file)
    else:
        # 实时摄像头模式
        print("📸 未提供照片，使用摄像头采集...")
        img_path = capture_face(user_dir, name)

    if img_path and face_auth.register_face(name, img_path):
        print(f"✅ 用户 {name} 注册成功！")

        # 将人脸编码存储到数据库
        encoding = face_auth.get_encoding(img_path)
        if encoding is not None:
            save_user(name, encoding.tobytes())

        return {"message": f"用户 {name} 注册成功"}
    raise HTTPException(status_code=400, detail="人脸注册失败")


# 人脸验证接口
@app.post("/verify")
async def verify(file: UploadFile = File(None)):
    """
    验证人脸：
    - 上传图片或通过摄像头实时验证
    - 自动遍历数据库
    """
    img_path = "./temp/temp_verify.jpg"

    if file:
        # 使用上传图片验证
        with open(img_path, "wb") as img_file:
            shutil.copyfileobj(file.file, img_file)
    else:
        # 实时摄像头验证
        print("📸 未提供照片，使用摄像头实时检测...")
        img_path = capture_face("./temp", "temp")

    # 遍历数据库进行验证
    users = load_users()
    for name, encoding_bytes in users:
        encoding = face_auth.bytes_to_encoding(encoding_bytes)
        matched, confidence = face_auth.compare_encoding(img_path, encoding)

        if matched:
            return {"status": "success", "user": name, "confidence": confidence}

    raise HTTPException(status_code=401, detail="人脸识别失败")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
