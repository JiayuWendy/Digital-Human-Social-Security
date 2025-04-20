import librosa
import numpy as np
import os
import speechbrain as sb
from speechbrain.speechbrain.inference.classifiers import EncoderClassifier
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyaudio
import wave
import requests
import noisereduce as nr
import cv2
import time
import pickle
import shutil
import uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from datetime import datetime

# python speaker-rec.py
THRESHOLD=0.6
device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化声纹模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}  # 新增设备设置
)

# 初始化Whisper语音识别
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

# 声纹+人脸记忆库
memory_db = {
}

# 人脸识别初始化
class FaceAuth:
    def __init__(self, encoding_dir="./face_encodings"):
        self.encoding_dir = encoding_dir
        os.makedirs(self.encoding_dir, exist_ok=True)

    def save_encoding(self, name, encoding):
        path = os.path.join(self.encoding_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(encoding, f)

    def load_encoding(self, name):
        path = os.path.join(self.encoding_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def load_all_encodings(self):
        encodings = {}
        for filename in os.listdir(self.encoding_dir):
            if filename.endswith(".pkl"):
                name = filename.split(".")[0]
                encodings[name] = self.load_encoding(name)
        return encodings

face_auth = FaceAuth()

def save_memory_db():
    """保存记忆库到磁盘"""
    for user_id, data in memory_db.items():
        np.save(f"voiceprints/{user_id}.npy", data["voiceprint"])
        with open(f"voiceprints/{user_id}_mem.txt", "w") as f:
            f.write("\n".join(data["memories"]))

def load_memory_db():
    """从磁盘加载记忆库"""
    if not os.path.exists("voiceprints"):
        return {}
    
    db = {}
    for file in os.listdir("voiceprints"):
        if file.endswith(".npy"):
            user_id = file.split(".")[0]
            voiceprint = np.load(f"voiceprints/{file}")
            mem_file = f"voiceprints/{user_id}_mem.txt"
            memories = []
            if os.path.exists(mem_file):
                with open(mem_file) as f:
                    memories = f.read().splitlines()
            db[user_id] = {
                "voiceprint": voiceprint,
                "memories": memories
            }
    return db

# 在初始化时加载已有记忆库
memory_db = load_memory_db()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def record_audio(duration=5, save_path="temp.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    print(f"录音中...（{duration}秒）")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存临时文件（实际应用可直接传内存数据）
    wf = wave.open(save_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return save_path

def extract_voiceprint(audio_path):
    # 加载时添加语音增强
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 语音增强（降噪）
    y = librosa.effects.preemphasis(y)  # 预加重
    y = nr.reduce_noise(y=y, sr=sr)    
    
    # 语音活性检测（去除静音段）
    intervals = librosa.effects.split(y, top_db=30)
    y_clean = np.concatenate([y[begin:end] for begin,end in intervals])
    
    # 标准化音频长度（至少3秒）
    if len(y_clean) < sr*3:
        y_clean = np.pad(y_clean, (0, max(0, sr*3 - len(y_clean))))
    
    # 输入模型
    y_tensor = torch.from_numpy(y_clean).float().unsqueeze(0).to(device)
    return classifier.encode_batch(y_tensor).squeeze().cpu().numpy()

def transcribe_audio(audio_path):
    """语音转文本"""
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)
    predicted_ids = asr_model.generate(
        inputs.input_features,
        forced_decoder_ids=forced_decoder_ids
    )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
'''
def register_user(audio_path):
    """注册新用户并创建记忆库"""
    voiceprint = extract_voiceprint(audio_path)
    user_id = f"user_{len(memory_db) + 1}"
    memory_db[user_id] = {"voiceprint": voiceprint, "memories": []}
    np.save(f"voiceprints/{user_id}.npy", voiceprint)
    return user_id
'''
def register_user(audio_path, face_image=None):
    """注册新用户并创建记忆库"""
    # 提取声纹
    voiceprint = extract_voiceprint(audio_path)
    user_id = f"user_{len(memory_db) + 1}"
    
    # 提取人脸特征
    face_encoding = None
    if face_image is not None:
        face_locations = face_recognition.face_locations(face_image)
        if face_locations:
            face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
            face_auth.save_encoding(user_id, face_encoding)
    
    # 存储用户信息
    memory_db[user_id] = {
        "voiceprint": voiceprint,
        "memories": [],
        "face_encoding": face_encoding
    }
    np.save(f"voiceprints/{user_id}.npy", voiceprint)
    return user_id

'''
def recognize_user(audio_path, threshold=THRESHOLD):
    """识别用户并返回对应记忆库"""
    current_voiceprint = extract_voiceprint(audio_path)
    for user_id, data in memory_db.items():
        similarity = cosine_similarity(current_voiceprint, data["voiceprint"])
        print(user_id,similarity)
        if similarity > threshold:
            return user_id, data["memories"]
    new_user_id = register_user(audio_path)
    return new_user_id, memory_db[new_user_id]["memories"]
'''

def recognize_user(audio_path, face_image=None, threshold=THRESHOLD):
    """多模态用户识别"""
    current_voiceprint = extract_voiceprint(audio_path)
    best_match = None
    max_sim = 0
    
    # 声纹比对
    for user_id, data in memory_db.items():
        voice_sim = cosine_similarity(current_voiceprint, data["voiceprint"])
        face_sim = 0
        
        # 人脸比对（如果提供）
        if face_image is not None and "face_encoding" in data and data["face_encoding"] is not None:
            face_locations = face_recognition.face_locations(face_image)
            if face_locations:
                current_face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                face_sim = 1 - face_recognition.face_distance([data["face_encoding"]], current_face_encoding)[0]
        
        # 综合相似度（取最高值）
        total_sim = max(voice_sim, face_sim)
        print(f"用户 {user_id} 相似度 - 声纹:{voice_sim:.2f} 人脸:{face_sim:.2f}")
        
        if total_sim > max_sim:
            max_sim = total_sim
            best_match = user_id
    
    # 动态阈值策略
    if max_sim > threshold:
        return best_match, memory_db[best_match]["memories"]
    return register_user(audio_path, face_image), []

def generate_deepseek_response(user_id, user_input, memories):
    """调用深度求索大模型API"""
    headers = {
        "Authorization": f"Bearer {'sk-38ebb5e38c864eb781c9d54df7d3d752'}",
        "Content-Type": "application/json"
    }
    
    # 构建符合深度求索API要求的请求体
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": f"[用户{user_id}历史记忆]\n" + "\n".join(memories[-3:]) if memories else "新用户对话"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return "当前服务不可用，请稍后再试"

def capture_face_image():
    """从摄像头捕获一帧人脸图像"""
    cap = cv2.VideoCapture(0)

    # 预热摄像头
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame).astype(np.uint8)  # ⭐ 添加类型转换
        return rgb_frame
    return None


'''   
def full_interaction(audio_path):
    """完整交互流程"""
    # 语音识别
    user_input = transcribe_audio(audio_path)
    print(f"识别文本: {user_input}")
    
    # 声纹识别
    user_id, memories = recognize_user(audio_path)
    
    # 生成回复（真实LLM调用）
    llm_response = generate_deepseek_response(user_id, user_input, memories)
    
    # 更新记忆
    memory_db[user_id]["memories"].extend([user_input, llm_response])
    
    return llm_response
'''
def full_interaction():
    """完整交互流程"""
    # 1. 录音
    audio_path = record_audio()
    
    # 2. 拍照
    print("请面对摄像头...")
    face_image = capture_face_image()
    face_image = capture_face_image()
    if face_image is None:
        raise ValueError("未成功从摄像头捕获人脸图像！请检查摄像头是否连接或被其他程序占用")
    
    # 3. 语音识别
    user_input = transcribe_audio(audio_path)
    print(f"识别文本: {user_input}")
    
    # 4. 多模态用户识别
    user_id, memories = recognize_user(audio_path, face_image)
    
    # 5. 生成回复
    llm_response = generate_deepseek_response(user_id, user_input, memories)
    
    # 6. 更新记忆
    memory_db[user_id]["memories"].extend([user_input, llm_response])
    
    return llm_response

if __name__ == "__main__":
    os.makedirs("voiceprints", exist_ok=True)
    os.makedirs("face_encodings", exist_ok=True)
    memory_db = load_memory_db()
    
    while True:
        try:
            cmd = input("\n按回车开始交互 (q退出): ").strip()
            if cmd.lower() == 'q':
                save_memory_db()
                print("系统退出")
                break
                
            response = full_interaction()
            print(f"\nAI回复: {response}")
            
        except KeyboardInterrupt:
            save_memory_db()
            break
        finally:
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")