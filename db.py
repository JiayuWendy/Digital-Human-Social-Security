import sqlite3

DB_FILE = "./data/faces.db"

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        face_encoding BLOB NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# 保存用户
def save_user(name, encoding):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)", (name, encoding))
    conn.commit()
    conn.close()

# 加载用户
def load_users():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_encoding FROM users")
    users = cursor.fetchall()
    conn.close()
    return users
