import os
import sqlite3
import requests
from flask import Flask, request, jsonify
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from waitress import serve
import json
from flask import Response

app = Flask(__name__)
SECRET_KEY = b"this_is_a_fixed_32_byte_key_123!"
DEEPSEEK_API_KEY = "sk-38ebb5e38c864eb781c9d54df7d3d752"
DB_DIR = "databases"
os.makedirs(DB_DIR, exist_ok=True)

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
            encrypted_content BLOB
        )
    """)
    conn.commit()
    return conn, cursor

def send_to_deepseek_api(message: str):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": message}]}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    return "对话失败，请稍后再试。"

def get_security_level(content: str):
    prompt = f"这个信息的敏感等级是1（普通）还是2（敏感）？请输出数字即可，不要回答其他内容。内容：{content}"
    response = send_to_deepseek_api(prompt)
    return 2 if "2" in response else 1

@app.route("/store", methods=["POST"])
def store_message():
    data = request.json
    user_id = data.get("user_id")
    content = data.get("content")
    if not all([user_id, content]):
        return jsonify({"error": "缺少参数"}), 400
    
    security_level = get_security_level(content)
    
    print(f"[DEBUG] 原始消息: {content}")
    encrypted_content = encrypt_data(content, SECRET_KEY)
    if not encrypted_content:
        print("[ERROR] 加密数据为空，存储失败")
        return jsonify({"error": "加密失败"}), 500

    print(f"[DEBUG] 加密后数据长度: {len(encrypted_content)}")
    
    conn, cursor = get_user_db(user_id)
    cursor.execute("INSERT INTO messages (user_message, security_level, encrypted_content) VALUES (?, ?, ?)",
                   (content, security_level, sqlite3.Binary(encrypted_content)))
    conn.commit()
    conn.close()
    return jsonify({"status": "stored", "security_level": security_level})

@app.route("/retrieve", methods=["POST"])
def retrieve_message():
    try:
        data = request.json
        user_id = data.get("user_id")
        access_level = data.get("access_level")

        if not all([user_id, access_level]):
            return jsonify({"error": "缺少参数"}), 400

        conn, cursor = get_user_db(user_id)
        cursor.execute("SELECT encrypted_content FROM messages WHERE security_level <= ?", (access_level,))
        records = cursor.fetchall()
        conn.close()

        messages = []
        for record in records:
            if record[0] is None:
                print("[WARNING] 发现空消息，跳过")
                continue  # 跳过空消息
            try:
                decrypted_message = decrypt_data(bytes(record[0]), SECRET_KEY)
                messages.append(decrypted_message)
            except Exception as e:
                print(f"[ERROR] 解密失败: {e}")
                messages.append("[解密失败的消息]")

        return Response(json.dumps({"messages": messages}, ensure_ascii=False), mimetype="application/json")


    except Exception as e:
        print(f"[ERROR] 服务器崩溃: {e}")
        return jsonify({"error": "服务器错误"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """与大模型进行对话，并结合历史记忆"""
    data = request.json
    user_id = data.get("user_id")
    message = data.get("message")
    access_level = data.get("access_level", 1)  # 默认安全等级为 1

    if not all([user_id, message]):
        return jsonify({"error": "缺少参数"}), 400

    conn, cursor = get_user_db(user_id)
    cursor.execute(
        "SELECT user_message, model_response FROM messages WHERE security_level <= ? ORDER BY id DESC LIMIT 5",
        (access_level,)
    )
    history_records = cursor.fetchall()
    conn.close()

    history_messages = []
    for user_msg, model_resp in history_records:
        history_messages.append(f"用户: {user_msg}\n助手: {model_resp}")

    history_text = "\n".join(history_messages)
    full_prompt = f"以下是你的聊天历史：\n{history_text}\n\n当前问题：{message}"
    model_response = send_to_deepseek_api(full_prompt)

    conn, cursor = get_user_db(user_id)
    try:
        security_level = get_security_level(message)
        encrypted_response = encrypt_data(model_response, SECRET_KEY)  # **加密 AI 响应**
        cursor.execute(
            "INSERT INTO messages (user_message, model_response, security_level, encrypted_content) VALUES (?, ?, ?, ?)",
            (message, model_response, security_level,sqlite3.Binary(encrypted_response))
        )
        conn.commit()
    finally:
        conn.close()

    return Response(json.dumps({"response": model_response}, ensure_ascii=False), mimetype="application/json")

if __name__ == "__main__":
    print("[INFO] 服务器启动: http://127.0.0.1:5000")
    serve(app, host="0.0.0.0", port=5000)

#从 SQLite 记忆库中按安全等级 查询最近 5 条聊天记录。
#格式化历史聊天记录（"用户: XX\n助手: XX"）。
#把历史聊天+当前问题拼接 成完整的 Prompt，发给 DeepSeek Chat。
#存储新的聊天记录 进数据库，方便下次检索。
#返回大模型回复（支持中文显示）

#现已实现：加密消息存储、解密检索、以及基于 SQLite 和 Waitress 的本地部署功能，基本满足了你对 消息加密存储和检索的需求。你可以通过接口进行消息的存储与检索，所有存储的消息都是加密的，只有拥有相应访问权限的用户才能查看相应的内容
# 1.实现输入消息并存储（以不同身份发送/store请求即可）。
# 2.不同用户的消息都要分开存储（为每个用户建立不同的.db文件）。
# 3.仔细研究不同等级如何检索信息（赋予每个消息不同的安全等级)
# 4.每次聊天先从历史消息库中检索最近的5条历史记录，然后一同喂给大模型

#task：分两个安全等级：1.单认证：level=1，2.双重认证：level=2. 大模型自动根据敏感级别将消息分类。