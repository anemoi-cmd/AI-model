import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import time
import spacy
import faiss
import numpy as np
import mysql.connector
from dashscope import Application
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
import dashscope

try:
    nlp = spacy.load("zh_core_web_sm")
except OSError:
    print("错误：未找到 spaCy 中文模型。请运行以下命令安装模型：")
    exit(1)

APP_ID = os.getenv("DASHSCOPE_APP_ID", "727767bdc7aa42d0adfdecb3137c9fd0")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-430eda3a5950475fa5dde7d12f51199f")

dashscope.api_key = API_KEY

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)

index_file_path = 'C:/Users/12393/Vscode/vector_index.index'

if not os.path.exists(index_file_path):
    print(f"错误：未找到 FAISS 索引文件，请确认路径：{index_file_path}")
    exit(1)

faiss_index = faiss.read_index(index_file_path)

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "76825917jy",
    "database": "my_database",
    "charset": "utf8mb4"
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    select_query = "SELECT id, key_name, value_text FROM store_knowledge ORDER BY id"
    cursor.execute(select_query)
    results = cursor.fetchall()
    conn.close()
except mysql.connector.Error as err:
    print(f"数据库连接错误：{err}")
    exit(1)

data = []
for row in results:
    data.append({
        "id": row[0], # type: ignore
        "key_name": row[1], # type: ignore
        "value_text": row[2] # type: ignore
    })

EXIT_KEYWORDS_FILE = "exit_keywords.txt"

def load_exit_keywords(file_path):
    if not os.path.exists(file_path):
        default_keywords = {
            "退出",
            "再见",
            "结束",
            "bye",
            "goodbye",
            "quit",
            "exit",
            "我问完了",
            "不用了",
            "没问题",
            "不用谢谢",
            "谢谢",
            "拜拜",
            "走了",
            "结束对话",
            "完成了",
            "不需要了",
            "先这样",
            "就这样吧",
            "好了",
            "不必了",
            "算了",
            "没事",
            "不需要",
        }
        save_exit_keywords(file_path, default_keywords)
        return default_keywords
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            keywords = set(line.strip() for line in f if line.strip())
        return keywords
    except IOError:
        return set()

def save_exit_keywords(file_path, keywords):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for keyword in sorted(keywords):
                f.write(f"{keyword}\n")
    except IOError:
        pass

exit_keywords = load_exit_keywords(EXIT_KEYWORDS_FILE)

EXIT_PATTERN = re.compile('|'.join([re.escape(keyword) for keyword in exit_keywords]), re.IGNORECASE)

def is_exit_command(user_input):
    if EXIT_PATTERN.search(user_input):
        return True
    return False

def stream_output(content, delay=0.02):
    for char in content:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

conversation_history = []
MAX_HISTORY_ROUNDS = 10

def truncate_conversation_history(history, max_rounds=MAX_HISTORY_ROUNDS):
    if len(history) > max_rounds * 2:
        history = history[-max_rounds * 2:]
    return history

def call_agent_app(user_input):
    try:
        query_embedding = embedder.get_text_embedding(user_input)
    except Exception as e:
        return f"嵌入模型调用失败：{str(e)}"

    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)

    k = 5
    distances, indices = faiss_index.search(query_vector, k)

    context_list = []
    for idx in indices[0]:
        if idx < len(data):
            item = data[idx]
            context_list.append(f"{item['key_name']}：{item['value_text']}")
    context = "\n".join(context_list)

    history = "\n".join(conversation_history[-MAX_HISTORY_ROUNDS*2:])
    prompt = f"{history}\n用户：{user_input}\n助手："

    prompt += f"\n\n请根据以下已知信息，回答用户的问题。如果无法从中得到答案，请告诉用户您不知道。\n\n已知信息：\n{context}\n\n请注意，您的回答应仅基于以上提供的信息，不要添加任何其他内容。"

    try:
        response = Application.call(
            app_id=APP_ID,
            api_key=API_KEY,
            prompt=prompt,
            temperature=0,
            max_tokens=150,
            stop=["用户：", "助手："]
        )
        if response.get("status_code") == 200: # type: ignore
            output = response.get("output", {}) # type: ignore
            assistant_response = output.get("text", "未能提取到助手的回复内容。")
            return assistant_response.strip()
        else:
            message = response.get("message", "无错误消息") # type: ignore
            return f"错误：{response.get('status_code')} - {message}" # type: ignore
    except Exception as e:
        return f"调用失败：{str(e)}"

def main():
    global conversation_history
    print("欢迎使用智能助手！输入 '退出' 或相关语义结束对话。")
    while True:
        user_input = input("你：").strip()
        if not user_input:
            continue
        if is_exit_command(user_input):
            print("助手：感谢您的咨询，再见！")
            break

        response = call_agent_app(user_input)
        if isinstance(response, str) and (response.startswith("错误") or response.startswith("调用失败") or response.startswith("嵌入模型调用失败")):
            print(f"助手：{response}")
            continue

        conversation_history.append(f"用户：{user_input}")
        conversation_history.append(f"助手：{response}")

        print("助手：", end="", flush=True)
        stream_output(response)

        conversation_history = truncate_conversation_history(conversation_history)

if __name__ == "__main__":
    main()
