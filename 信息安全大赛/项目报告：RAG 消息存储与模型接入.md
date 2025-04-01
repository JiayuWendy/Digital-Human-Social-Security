# 项目报告：RAG 消息存储与模型接入

## 1. 项目概述

本项目基于 Flask 搭建一个具备 RAG（Retrieval-Augmented Generation）存储与 DeepSeek 大模型对接的聊天系统。该系统能够对用户消息进行存储、加密、分类，并结合历史消息与大模型进行交互。

## 2. 主要功能模块

### 2.1 RAG 消息存储

- **数据库管理（SQLite）**
  - 每个用户拥有独立的 SQLite 数据库。
  - 表 `messages` 存储用户输入、模型回复、安全等级和加密数据。
- **加密存储（AES-GCM）**
  - 使用固定密钥 `SECRET_KEY` 进行 AES-GCM 加密。
  - 仅存储密文，防止未授权访问。
- **安全等级分级**
  - 使用 DeepSeek 进行敏感性分析，自动判定消息的安全等级（1 级或 2 级）。
  - 1 级（普通）：所有用户可访问。
  - 2 级（敏感）：仅双重认证用户可访问。

### 2.2 模型接入（DeepSeek API）

- **消息处理**
  - 结合最新 5 条安全等级符合的聊天记录，构造完整上下文。
  - 发送至 DeepSeek 大模型获取智能回复。
- **安全等级自动识别**
  - 每条用户输入均由 DeepSeek 评估敏感程度。
  - 根据敏感性自动决定存储时的安全级别。

## 3. API 设计与测试方法

### 3.1 消息存储 API

**接口：** `/store`

- **方法：** `POST`

- **参数：** `user_id`, `content`

- **返回值：** `status`, `security_level`

- **测试方法：**

  ```bash
  curl -X POST "http://127.0.0.1:5000/store" -H "Content-Type: application/json" -d "{\"user_id\": \"test_user\", \"content\": \"我的银行卡号是1234\"}"
  ```

### 3.2 消息检索 API

**接口：** `/retrieve`

- **方法：** `POST`

- **参数：** `user_id`, `access_level`

- **返回值：** 解密后的 `messages`

- **测试方法：**

  ```bash
  curl -X POST "http://127.0.0.1:5000/retrieve" -H "Content-Type: application/json" -d "{\"user_id\": \"test_user\", \"access_level\": 2}"
  ```

### 3.3 对话 API

**接口：** `/chat`

- **方法：** `POST`

- **参数：** `user_id`, `message`, `access_level`

- **返回值：** `response`（模型回复）

- **测试方法：**

  ```bash
  curl -X POST "http://127.0.0.1:5000/chat" -H "Content-Type: application/json" -d "{\"user_id\": \"test_user\", \"message\": \"我在中国科学技术大学上学\"}"
  ```

## 4. 代码结构

- **`encrypt_data` / `decrypt_data`**: AES 加密与解密。
- **`get_user_db`**: SQLite 数据库管理。
- **`send_to_deepseek_api`**: 与 DeepSeek 交互。
- **`get_security_level`**: 自动识别敏感等级。
- **API 端点**: `/store`、`/retrieve`、`/chat`

## 5. 运行方式

```bash
python app.py
```

或者使用 `Waitress` 生产环境部署。

## 6. 未来优化方向

- **更细粒度的安全等级划分**

- **优化聊天历史上下文管理**

- **支持更多认证方式（如声纹识别）**

  **单一认证（声纹 或 面容）** → `access_level=1`，只能访问安全等级 ≤ 1 的消息。

  **双重认证（声纹 + 面容）** → `access_level=2`，可以访问安全等级 ≤ 2 的消息。

## 7.记忆库存储

本模型采用本地的数据库**SQLite**，无需接入API，现已实现：加密消息存储、解密检索、以及基于 SQLite 和 Waitress 的本地部署功能，满足消息加密存储和检索的需求。所有存储的消息都是加密的，只有拥有相应访问权限的用户才能查看相应的内容。

 1.实现输入消息并存储（以不同身份发送/store请求即可）。

 2.不同用户的消息都要分开存储（为每个用户建立不同的.db文件）。

 3.仔细研究不同等级如何检索信息（赋予每个消息不同的安全等级)

 4.每次聊天先从历史消息库中检索最近的5条历史记录，然后一同喂给大模型

## 8.记忆库等级机制

安全等级分为两个等级：1.单认证：level=1，2.双重认证：level=2 

**大模型自动根据敏感级别将消息分类**

修改 `/store` 路由，使其通过 DeepSeek 自动判断消息的敏感程度，并为消息分配 `security_level`。修改后的逻辑如下：

1. **在存储消息前，调用 DeepSeek 让其判断安全等级**：
   - 如果 DeepSeek 认为内容较为敏感，则设为 `security_level=2`。
   - 否则设为 `security_level=1`。
2. **删除 `store_message` 的 `security_level` 手动输入**：
   - 用户只需要传 `user_id` 和 `content`，系统会自动分配安全等级。
3. **新增 DeepSeek API 解析安全等级的功能**：
   - 通过 DeepSeek API 发送 `"这个信息的敏感等级是1（普通）还是2（敏感）？请输出数字即可，不要回答其他内容。内容：" + 用户输入的消息`。
   - 根据返回的 `1` 或 `2` 进行存储。



**记忆等级查询：（开发者需要查询时）**

1.下载SQLite数据库并解压。

2.打开 `cmd` 或 PowerShell，进入解压后的文件夹，输入：sqlite3（或者直接输入sqlite3的绝对路径，Eg："D:\Tools\sqlite-tools-win-x64-3490100\sqlite3.exe"）。

3.使用.open命令打开数据库（Eg：.open "D:/信息安全大赛/databases/test_user.db"）

4.使用命令查询（Eg：SELECT id, user_message, security_level FROM messages ORDER BY id DESC LIMIT 5;）

5.字段含义：

- **第一列（id）**：消息的唯一标识符（自增主键）。
- **第二列（user_message）**：用户输入的消息内容。
- **第三列（security_level）**：消息的安全等级（1 代表较低的安全等级）。

效果如下：

![d6fbd54b23d606c4cf0e667eb4d014a4](D:\QQ\Tencent Files\3423987604\nt_qq\nt_data\Pic\2025-03\Ori\d6fbd54b23d606c4cf0e667eb4d014a4.png)



**tips**:目前正在自费连接deep seek测试，有需要API可联系我







