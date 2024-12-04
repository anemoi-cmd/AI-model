from flask import Flask, request, jsonify
import os
import dashscope
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
import faiss
import mysql.connector
import numpy as np
import logging
from dashscope import Application
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 类：配置
class Config: 
    def __init__(self):
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-430eda3a5950475fa5dde7d12f51199f")
        self.DASHSCOPE_APP_ID = os.getenv("DASHSCOPE_APP_ID", "727767bdc7aa42d0adfdecb3137c9fd0")
        self.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "C:/Users/12393/Vscode/vector_index.index")
        self.DB_CONFIG = {
            "host": "localhost",
            "user": "root",
            "password": "76825917jy",
            "database": "my_database",
            "charset": "utf8mb4"
        }
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("app.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()


# 类：数据库        
class Database:
    def __init__(self, config: Config):
        self.config = config
        self.logger = config.logger
        self.connection = self.connect()
        
    def connect(self):
        try:
            conn = mysql.connector.connect(**self.config.DB_CONFIG)
            self.logger.info("成功连接到MySQL数据库。")
            return conn
        except mysql.connector.Error as err:
            self.logger.error(f"数据库连接错误：{err}")
            raise
        
    def get_data(self, ids):
        cursor = self.connection.cursor()
        placeholders = ','.join(['%s'] * len(ids))
        query = f"SELECT id, key_name, value_text FROM store_knowledge WHERE id IN ({placeholders})"
        try:
            cursor.execute(query, ids)
            results = cursor.fetchall()
            self.logger.info(f"成功从数据库中获取数据：{results}")
            return results
        except mysql.connector.Error as err:
            self.logger.error(f"数据库查询错误：{err}")
            return []
        finally:
            cursor.close()
    
    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            self.logger.info("数据库连接已关闭。")


# 类：向量转化存储
class Vector:
    def __init__(self, index_path: str, logger):
        self.logger = logger
        if not os.path.exists(index_path):
            self.logger.error(f"未找到FAISS索引文件，请确认路径：{index_path}")
            raise FileNotFoundError(f"未找到FAISS索引文件，请确认路径：{index_path}")
        try:
            self.index = faiss.read_index(index_path)
            self.logger.info(f"成功加载FAISS索引：{index_path}")
        except Exception as e:
            self.logger.error(f"加载FAISS索引失败：{e}")
            raise

    def search(self, query_vector, k=5):
        try:
            distances, indices = self.index.search(query_vector, k)
            self.logger.info(f"FAISS检索成功，返回距离和索引：{distances}, {indices}")
            return distances, indices
        except Exception as e:
            self.logger.error(f"FAISS检索失败：{e}")
            return [], []


# 类：嵌入模型
class Embedding:
    def __init__(self, api_key: str, logger):
        self.logger = logger
        os.environ["DASHSCOPE_API_KEY"] = api_key  # 设置环境变量
        try:
            self.embedder = DashScopeEmbedding(
                model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
                text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
            )
            self.logger.info("成功初始化嵌入模型。")
        except Exception as e:
            self.logger.error(f"初始化嵌入模型失败：{e}")
            raise
            
    def get_embedding(self, text: str):
        try:
            embedding = self.embedder.get_text_embedding(text)
            self.logger.info("成功获取文本嵌入。")
            return np.array(embedding).astype('float32').reshape(1, -1)
        except Exception as e:
            self.logger.error(f"嵌入模型调用失败：{e}")
            raise


# 类：调用大型语言模型（LLM）
class LLM:
    def __init__(self, app_id: str, api_key: str, logger):
        self.app_id = app_id
        self.api_key = api_key
        self.logger = logger
    
    def call_llm(self, prompt: str):
        try:
            response = Application.call(
                app_id=self.app_id,
                api_key=self.api_key,
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                stop=["用户：", "助手："]
            )
            if response.get("status_code") == 200:
                output = response.get("output", {})
                assistant_response = output.get("text", "").strip()
                self.logger.info("成功调用LLM生成回复。")
                return assistant_response
            else:
                message = response.get("message", "无错误消息")
                self.logger.error(f"LLM调用错误：{response.get('status_code')} - {message}")
                return f"错误：{response.get('status_code')} - {message}"
        except Exception as e:
            self.logger.error(f"调用LLM失败：{e}")
            return f"调用失败：{str(e)}"


# 类：助手机器人
class Chatbot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = config.logger
        self.db_connector = Database(config)
        self.vector = Vector(config.FAISS_INDEX_PATH, self.logger)
        self.embedding = Embedding(config.DASHSCOPE_API_KEY, self.logger)
        self.llm = LLM(config.DASHSCOPE_APP_ID, config.DASHSCOPE_API_KEY, self.logger)
        self.conversation_history = []
        self.MAX_HISTORY_ROUNDS = 10
    
    def truncate_conversation_history(self):
        if len(self.conversation_history) > self.MAX_HISTORY_ROUNDS * 2:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_ROUNDS * 2:]
            self.logger.info("截断对话历史以保持对话轮数。")
                
    def query(self, user_input: str):
        self.logger.info(f"处理用户查询：{user_input}")
        try:
            query_vector = self.embedding.get_embedding(user_input)
        except Exception as e:
            self.logger.error(f"嵌入模型调用失败：{e}")
            return "嵌入模型调用失败，请稍后再试。"
        
        distances, indices = self.vector.search(query_vector, k=5)
        
        ids = [int(idx) for idx in indices[0] if idx >= 0]
        if not ids:
            self.logger.warning("未找到相关信息。")
            return "未找到相关信息。"
        
        results = self.db_connector.get_data(ids)
        if not results:
            self.logger.warning("根据检索结果未从数据库获取到数据。")
            return "未找到相关信息。"
        
        context_list = [f"{row[1]}：{row[2]}" for row in results]  # type: ignore
        context = "\n".join(context_list)
        self.logger.info(f"组装上下文信息：\n{context}")
        
        history = "\n".join(self.conversation_history[-self.MAX_HISTORY_ROUNDS * 2:])
        prompt = f"{history}\n用户：{user_input}\n助手：\n已知以下信息：\n{context}\n请根据以上信息回答用户的问题。"
        self.logger.info(f"构建Prompt：\n{prompt}")
        
        assistant_response = self.llm.call_llm(prompt)
        self.logger.info(f"LLM回复：{assistant_response}")
        
        self.conversation_history.append(f"用户：{user_input}")
        self.conversation_history.append(f"助手：{assistant_response}")
        self.truncate_conversation_history()
        
        return assistant_response
    
    def close(self):
        self.db_connector.close()


# 类：Flask应用
class FlaskApp:
    def __init__(self, chatbot: Chatbot):
        self.app = Flask(__name__)
        self.chatbot = chatbot
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({'message': '欢迎使用聊天机器人API。请使用 /chat 路由发送POST请求。'})
    
        @self.app.route('/chat', methods=['POST'])
        def chat():
            data = request.get_json()
            user_input = data.get('message', '').strip()
            if not user_input:
                return jsonify({'response': '请输入您的问题。'})
            try:
                assistant_response = self.chatbot.query(user_input)
                return jsonify({'response': assistant_response})
            except Exception as e:
                self.chatbot.logger.error(f"服务器内部错误：{e}")
                return jsonify({'response': '服务器内部错误，请稍后再试。'}), 500
    
    def run(self, debug=True):
        self.app.run(debug=debug)


# 主程序
def main():
    config = Config()
    
    chatbot = Chatbot(config)
    
    flask_app = FlaskApp(chatbot)
    
    try:
        flask_app.run(debug=True)
    finally:
        chatbot.close()

if __name__ == '__main__':
    main()
