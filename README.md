# 人脸认证系统

本项目实现了一个基于 FastAPI 的实时人脸认证系统，利用 OpenCV 捕获视频流、使用 Face Recognition 提取人脸特征，并通过 PIL 绘制中文提示。系统主要支持以下两个功能：

- **注册人脸**  
  用户可以通过上传图片或使用摄像头采集多角度人脸图像进行注册，系统会将采集到的所有图像训练后生成一个平均特征向量模型，并将其存储到本地（编码文件存放在 `encodings/` 目录中）。

- **实时认证**  
  系统通过实时摄像头检测人脸，并遍历存储的编码文件，选出匹配度最高的结果。仅当匹配度超过 70% 时，认证成功；否则返回认证失败信息。认证过程中，视频窗口中会以中文显示检测状态和识别结果。

## 项目结构

```
/face_auth
│
├── data/                      # 存储采集的用户人脸图像（按用户名分目录存放）
│   ├── Alice/
│   └── Bob/
│
├── encodings/                 # 存储人脸特征编码文件 (.pkl)
│   ├── Alice.pkl
│   └── Bob.pkl
│
├── fonts/                     # 存储下载的中文字体文件（如 msyh.ttc）
│
├── main.py                    # FastAPI 服务入口（提供注册和认证接口）
├── face_recog.py              # 人脸识别与模型训练模块
├── video_stream.py            # 摄像头采集与实时认证模块
├── font.py                    # 字体下载与中文显示工具
├── requirements.txt           # 项目依赖包
└── README.md                  # 本文档
```

## 功能说明

- **注册人脸**  
  - 用户可以通过上传图片或使用摄像头采集人脸数据进行注册。  
  - 系统会对指定用户文件夹中的所有图像进行批量训练，计算平均特征向量，生成模型并保存在 `encodings/` 目录中。

- **实时认证**  
  - 系统通过摄像头持续检测人脸，并与本地存储的编码进行比对。  
  - 在检测周期内只返回匹配度最高的结果，当匹配度大于 70% 时认证成功，否则返回认证失败。

- **中文提示显示**  
  - 使用 PIL 和自定义字体自动下载功能，在视频窗口中绘制中文提示和标签，确保中文显示正常。

## 安装步骤

1. **克隆仓库**  
   ```bash
   git clone https://github.com/yourusername/face_auth.git
   cd face_auth
   ```

2. **创建虚拟环境**  
   ```bash
   python -m venv .venv
   # Linux/MacOS:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   ```

3. **安装依赖包**  
   ```bash
   pip install -r requirements.txt
   ```

4. **字体设置**  
   本系统使用中文字体支持，默认字体文件名称为 `msyh.ttc`。  
   - 若项目中未检测到该字体文件，系统会自动尝试从开源项目下载并解压字体文件到 `fonts/` 目录。  
   - 如自动下载失败，请手动下载 [msyh 压缩包](https://github.com/CroesusSo/msyh.git) 后解压，将字体文件放入 `fonts/` 目录，并确保文件名为 `msyh.ttc`。

## 使用方法

### 启动 FastAPI 服务

在项目根目录下运行：

```bash
uvicorn main:app --reload
```

服务启动后，访问 [http://0.0.0.0:8000](http://0.0.0.0:8000) 查看自动生成的 API 文档。

### API 接口

#### 1. 注册人脸

- **接口路径：** `POST /register`
- **功能：**  
  - 支持通过上传图片或使用摄像头采集人脸数据进行注册。  
  - 注册后会对用户文件夹中所有图像进行模型训练，生成特征编码文件存储在 `encodings/` 目录中。
- **示例请求：**
  - **上传图片注册：**
    - URL: `http://localhost:8000/register?name=Alice`
    - 在 Postman 中选择 `form-data`，添加键 `file` 并上传图片文件。
  - **摄像头采集注册：**
    - URL: `http://localhost:8000/register?name=Alice`
    - 不上传文件，系统会自动调用摄像头采集人脸图像。

#### 2. 实时认证

- **接口路径：** `POST /verify`
- **功能：**
  - 系统自动调用摄像头检测人脸，并遍历存储的编码文件，比对返回匹配度最高的结果。  
  - 只有当匹配度高于 70% 时，认证成功；否则返回认证失败提示。
- **示例请求：**
  - URL: `http://localhost:8000/verify`
  - 可通过 Postman 调用该接口进行实时认证。

## 调试与日志输出

- 控制台将输出摄像头检测、采集进度以及认证过程中的中文提示信息，确保调试时可以看到详细的日志信息。
- 如果中文显示为乱码，请检查字体文件是否存在或自动下载功能是否正常工作。

## 依赖项

主要依赖包包括：
- [FastAPI](https://fastapi.tiangolo.com/) – 构建 REST API 服务
- [Uvicorn](https://www.uvicorn.org/) – ASGI 服务器
- [OpenCV](https://opencv.org/) – 图像和视频处理
- [Face Recognition](https://github.com/ageitgey/face_recognition) – 人脸识别库
- [Pillow](https://python-pillow.org/) – 图像处理及中文文本绘制
- [NumPy](https://numpy.org/) – 数值计算
- [Pickle](https://docs.python.org/3/library/pickle.html) – 序列化数据

## 授权

本项目采用 MIT 许可协议。详细内容请参见 LICENSE 文件。

## 致谢

- [Face Recognition](https://github.com/ageitgey/face_recognition)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- 开源中文字体项目：[msyh](https://github.com/CroesusSo/msyh.git)
