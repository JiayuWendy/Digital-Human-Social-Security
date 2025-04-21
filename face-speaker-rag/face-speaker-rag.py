import librosa
import numpy as np
import os
import torch
from speechbrain.inference.classifiers import EncoderClassifier
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(
    source="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyaudio
import wave
import requests
import noisereduce as nr
import time
import pickle
import shutil
import uvicorn
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from datetime import datetime
import cv2
import sqlite3
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# 初始化配置
THRESHOLD = 0.6
SECRET_KEY = b"this_is_a_fixed_32_byte_key_123!"
DEEPSEEK_API_KEY = "sk-38ebb5e38c864eb781c9d54df7d3d752"
DB_DIR = "databases"
os.makedirs(DB_DIR, exist_ok=True)  # 修复这里，去掉括号


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

# 人脸识别模块
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

# 加密解密函数
def encrypt_data(data: str, key: bytes) -> bytes:
    iv = os.urandom(12)
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data.encode('utf-8')) + encryptor.finalize()
    return iv + encryptor.tag + ciphertext

def decrypt_data(encrypted_data: bytes, key: bytes) -> str:
    try:
        if len(encrypted_data) < 28:
            raise ValueError("Encrypted data too short!")
        iv, tag, ciphertext = encrypted_data[:12], encrypted_data[12:28], encrypted_data[28:]
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        return (decryptor.update(ciphertext) + decryptor.finalize()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] 解密失败: {e}")
        return "[无法解密的消息]"

# 数据库操作
def get_user_db(user_id: str):
    db_path = os.path.join(DB_DIR, f"{user_id}.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            model_response TEXT,
            security_level INTEGER,
            encrypted_content BLOB,
            auth_method TEXT
        )
    """)

    conn.commit()
    return conn, cursor

# 音频处理函数
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
    
    wf = wave.open(save_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return save_path

def extract_voiceprint(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.effects.preemphasis(y)
    y = nr.reduce_noise(y=y, sr=sr)    
    intervals = librosa.effects.split(y, top_db=30)
    y_clean = np.concatenate([y[begin:end] for begin,end in intervals])
    if len(y_clean) < sr*3:
        y_clean = np.pad(y_clean, (0, max(0, sr*3 - len(y_clean))))
    y_tensor = torch.from_numpy(y_clean).float().unsqueeze(0).to(device)
    return classifier.encode_batch(y_tensor).squeeze().cpu().numpy()

def transcribe_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(
        y, 
        sampling_rate=sr, 
        return_tensors="pt",
        return_attention_mask=True  # 添加这一行
    ).to(device)
    predicted_ids = asr_model.generate(
        inputs.input_features,
        forced_decoder_ids=forced_decoder_ids
    )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# 人脸捕获
def capture_face_image():
    cap = cv2.VideoCapture(0)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame).astype(np.uint8)
        return rgb_frame
    return None

# 用户注册与识别
def register_user(audio_path, face_image=None):
    voiceprint = extract_voiceprint(audio_path)
    user_id = f"user_{len([f for f in os.listdir(DB_DIR) if f.endswith('.db')]) + 1}"
    
    # 存储声纹特征（带标记）
    voice_data = f"[声纹特征]{voiceprint.tobytes().decode('latin1')}"
    
    conn, cursor = get_user_db(user_id)
    encrypted = encrypt_data(voice_data, SECRET_KEY)
    cursor.execute(
        "INSERT INTO messages (user_message, model_response, security_level, encrypted_content, auth_method) VALUES (?, ?, ?, ?, ?)",
        ("[声纹注册]", "[系统消息]", 1, sqlite3.Binary(encrypted), "both")
    )
    conn.commit()
    conn.close()
    
    # 存储人脸特征
    if face_image is not None:
        face_locations = face_recognition.face_locations(face_image)
        if face_locations:
            face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
            face_auth.save_encoding(user_id, face_encoding)
    
    return user_id

def recognize_user(audio_path, face_image=None, threshold=THRESHOLD):
    current_voiceprint = extract_voiceprint(audio_path)
    best_match = None
    max_sim = 0
    auth_method = "voice"
    
    for db_file in os.listdir(DB_DIR):
        if not db_file.endswith(".db"):
            continue
            
        user_id = db_file.split(".")[0]
        conn, cursor = get_user_db(user_id)
        
        # 获取第一条消息作为声纹注册样本
        cursor.execute("SELECT encrypted_content FROM messages ORDER BY id ASC LIMIT 1")
        voice_data = cursor.fetchone()
        conn.close()
        
        if not voice_data:
            continue
            
        try:
            # 解密数据
            decrypted = decrypt_data(bytes(voice_data[0]), SECRET_KEY)
            if not decrypted or not decrypted.startswith("[声纹特征]"):
                continue
                
            # 提取存储的声纹特征
            stored_voiceprint = np.frombuffer(
                decrypted.split("\n")[0][len("[声纹特征]"):].encode('latin1'),
                dtype=np.float32
            )
            
            # 确保向量长度一致
            min_len = min(len(current_voiceprint), len(stored_voiceprint))
            if min_len == 0:
                continue
                
            voice_sim = cosine_similarity(
                current_voiceprint[:min_len], 
                stored_voiceprint[:min_len]
            )
        except Exception as e:
            print(f"Error processing voiceprint for {user_id}: {e}")
            continue
            
        face_sim = 0
        if face_image is not None:
            face_encoding = face_auth.load_encoding(user_id)
            if face_encoding is not None:
                face_locations = face_recognition.face_locations(face_image)
                if face_locations:
                    current_face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                    face_sim = 1 - face_recognition.face_distance([face_encoding], current_face_encoding)[0]
        
        total_sim = max(voice_sim, face_sim)
        print(f"用户 {user_id} 相似度 - 声纹:{voice_sim:.2f} 人脸:{face_sim:.2f}")
        
        if total_sim > max_sim:
            max_sim = total_sim
            best_match = user_id
            auth_method = "both" if face_sim > threshold and voice_sim > threshold else "voice"
    
    if max_sim > threshold:
        return best_match, auth_method
    return register_user(audio_path, face_image), "both"

# 大模型交互
def get_security_level(content):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "system",
            "content": "只输出数字1或2，不要其他内容。1表示普通信息，2表示敏感信息。"
        }, {
            "role": "user", 
            "content": f"这个信息的敏感等级是？内容：{content}"
        }]
    }
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", json=data, headers=headers)
        return 2 if "2" in response.json()['choices'][0]['message']['content'] else 1
    except:
        return 1

def chat_with_llm(user_id, user_input, auth_method):
    conn, cursor = get_user_db(user_id)
    
    # 根据认证方式确定安全等级
    access_level = 2 if auth_method == "both" else 1
    cursor.execute(
        "SELECT user_message, model_response FROM messages WHERE security_level <= ? ORDER BY id DESC LIMIT 5",
        (access_level,)
    )
    history_records = cursor.fetchall()
    
    # 构建历史记录
    history_messages = []
    for user_msg, model_resp in history_records:
        history_messages.append(f"用户: {user_msg}\n助手: {model_resp}")
    history_text = "\n".join(history_messages)
    
    # 调用大模型
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "system",
            "content": f"历史对话:\n{history_text}\n\n请回答用户最新问题"
        }, {
            "role": "user",
            "content": user_input
        }]
    }
    
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", json=data, headers=headers)
        model_response = response.json()['choices'][0]['message']['content']
        
        # 存储对话
        security_level = get_security_level(user_input)
        encrypted_content = encrypt_data(f"{user_input}\n{model_response}", SECRET_KEY)
        cursor.execute(
            "INSERT INTO messages (user_message, model_response, security_level, encrypted_content, auth_method) VALUES (?, ?, ?, ?, ?)",
            (user_input, model_response, security_level, sqlite3.Binary(encrypted_content), auth_method)
        )
        conn.commit()
        return model_response
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return "当前服务不可用，请稍后再试"
    finally:
        conn.close()

# 主交互流程
def full_interaction():
    # 1. 录音
    audio_path = record_audio()
    
    # 2. 拍照
    print("请面对摄像头...")
    face_image = capture_face_image()
    if face_image is None:
        raise ValueError("未成功从摄像头捕获人脸图像！")
    
    # 3. 语音识别
    user_input = transcribe_audio(audio_path)
    print(f"识别文本: {user_input}")
    
    # 4. 多模态用户识别
    user_id, auth_method = recognize_user(audio_path, face_image)
    print(f"认证方式: {auth_method}")
    
    # 5. 生成回复
    response = chat_with_llm(user_id, user_input, auth_method)
    return response

if __name__ == "__main__":
    os.makedirs("voiceprints", exist_ok=True)
    os.makedirs("face_encodings", exist_ok=True)
    
    while True:
        try:
            cmd = input("\n按回车开始交互 (q退出): ").strip()
            if cmd.lower() == 'q':
                print("系统退出")
                break
                
            response = full_interaction()
            print(f"\nAI回复: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
        finally:
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")