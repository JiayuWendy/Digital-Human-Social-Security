"""
main.py

FastAPI服务：
- 注册人脸：上传图片或使用摄像头采集
- 实时认证：摄像头检测当前用户
- 人脸特征编码存储在本地文件夹
"""
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from video_stream import capture_face_motion, recognize_camera
from face_recog import FaceAuth

app = FastAPI()

# 初始化人脸识别类（指定编码存储目录）
ENCODING_DIR = "./encodings"
face_auth = FaceAuth(encoding_dir=ENCODING_DIR)


# ✅ 注册人脸接口
@app.post("/register")
async def register(name: str, file: UploadFile = File(None)):
    """
    注册人脸：
    - 将编码存储到本地 .pkl 文件
    """
    user_dir = f"./data/{name}"
    os.makedirs(user_dir, exist_ok=True)

    if file:
        # ✅ 上传图片模式
        img_path = f"{user_dir}/{name}.jpg"
        with open(img_path, "wb") as img_file:
            shutil.copyfileobj(file.file, img_file)

    else:
        # ✅ 实时摄像头模式
        print("📸 未提供照片，使用摄像头采集...")
        img_paths = capture_face_motion(user_dir, name)

        if not img_paths:
            raise HTTPException(status_code=500, detail="摄像头采集失败")

    # ✅ 一次性对整个文件夹进行模型训练
    if face_auth.train_model_from_folder(user_dir, name):
        return {"message": f"用户 {name} 注册成功，模型已保存"}
    else:
        raise HTTPException(status_code=500, detail="人脸注册失败")


# ✅ 实时认证接口
@app.post("/verify")
async def verify():
    """
    实时认证：
    - 自动检测摄像头
    - 检测人脸与编码比对
    - 超时返回失败提示
    """
    print("🔍 开始实时认证...")

    result = recognize_camera(encoding_dir=ENCODING_DIR, timeout=120)

    if result:
        user, confidence = result
        print(f"✅ 认证成功: 用户={user}, 置信度={confidence:.2f}")
        return JSONResponse(content={
            "status": "success",
            "user": user,
            "confidence": f"{confidence:.2f}"
        })

    # 超时或失败返回
    print("❌ 认证失败或超时")
    return JSONResponse(content={
        "status": "failed",
        "message": "认证失败或超时"
    })


# ✅ 运行 FastAPI 服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
