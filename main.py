"""
FastAPI服务
人脸识别模块

功能说明：
    -可集成到RAG系统的前端
    -前端调用 /register 接口，录入人脸
    -每次对话前调用 /verify，验证人脸身份
    -验证成功后加载用户对应的 RAG 记忆库

使用说明：
    运行 FastAPI 服务：
    -uvicorn main:app --reload

    使用 Postman 测试：
    -/register 注册人脸：
    POST http://localhost:8000/register?name=Alice 上传图片：`face1.jpg`
    -/verify 验证人脸
    POST http://localhost:8000/verify 上传图片：`face2.jpg`
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from face_recognition import FaceAuth
import uvicorn
import shutil
import os

app = FastAPI()
face_auth = FaceAuth()


# 人脸注册接口
@app.post("/register")
async def register(name: str, file: UploadFile = File(...)):
    img_path = f"./data/{name}.jpg"

    # 保存图片
    with open(img_path, "wb") as img_file:
        shutil.copyfileobj(file.file, img_file)

    # 注册人脸
    if face_auth.register_face(name, img_path):
        return {"message": f"User {name} registered successfully"}
    raise HTTPException(status_code=400, detail="Face registration failed")


# 人脸认证接口
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    img_path = "./temp_verify.jpg"

    # 保存临时图片
    with open(img_path, "wb") as img_file:
        shutil.copyfileobj(file.file, img_file)

    # 验证人脸
    name, confidence = face_auth.verify_face(img_path)

    if name:
        return {"status": "success", "user": name, "confidence": confidence}

    raise HTTPException(status_code=401, detail="Face not recognized")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
