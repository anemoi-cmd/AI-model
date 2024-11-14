import os
import re
from dashscope import Application
import spacy
import time

try:
    nlp = spacy.load("zh_core_web_sm")
except OSError:
    print("错误：未找到spaCy中文模型。请运行以下命令安装模型：")
    print("python -m spacy download zh_core_web_sm")
    exit(1)

APP_ID = os.getenv("ALIYUN_APP_ID", "727767bdc7aa42d0adfdecb3137c9fd0")
API_KEY = os.getenv("ALIYUN_API_KEY", "sk-430eda3a5950475fa5dde7d12f51199f")

if not APP_ID or not API_KEY:
    print("错误：未设置应用ID或API Key，请配置环境变量。")
    exit(1)

conversation_history = []

MAX_HISTORY_ROUNDS = 10

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
exit_docs = [nlp(keyword) for keyword in exit_keywords]
SIMILARITY_THRESHOLD = 0.65


def truncate_conversation_history(history, max_rounds=MAX_HISTORY_ROUNDS):
    if len(history) > max_rounds * 2:
        history = history[-max_rounds * 2:]
    return history


def stream_output(content, delay=0.02):
    words = content.split()
    for word in words:
        print(word, end=" ", flush=True)
        time.sleep(delay)
    print()


def call_agent_app(prompt):
    try:
        response = Application.call(app_id=APP_ID, api_key=API_KEY, prompt=prompt)
        if response.get("status_code") == 200: # type: ignore
            output = response.get("output", {}) # type: ignore
            assistant_response = output.get("text", "未能提取到助手的回复内容。")
            return assistant_response
        else:
            message = response.get("message", "无错误消息") # type: ignore
            return f"错误：{response.get('status_code')} - {message}" # type: ignore
    except Exception as e:
        return f"调用失败：{str(e)}"


def is_exit_command(user_input):
    global exit_keywords, exit_docs, EXIT_PATTERN
    if EXIT_PATTERN.search(user_input):
        return True
    user_doc = nlp(user_input)
    for exit_doc in exit_docs:
        similarity = user_doc.similarity(exit_doc)
        if similarity >= SIMILARITY_THRESHOLD:
            confirmation = input(f'检测到退出命令意图，是否将 "{user_input}" 添加到退出关键词中？ (y/n): ').strip().lower()
            if confirmation == "y":
                exit_keywords.add(user_input)
                exit_docs.append(nlp(user_input))
                save_exit_keywords(EXIT_KEYWORDS_FILE, exit_keywords)
                EXIT_PATTERN = re.compile("|".join([re.escape(keyword) for keyword in exit_keywords]), re.IGNORECASE)
                print(f"新增退出关键词: {user_input}")
            return True
    return False


def main():
    global conversation_history
    print("欢迎使用大模型对话系统！输入 '退出' 或相关语义结束对话。")
    while True:
        user_input = input("你：").strip()
        if not user_input:
            continue
        if is_exit_command(user_input):
            print("助手：感谢您的咨询，再见！")
            break
        conversation_history.append(f"用户：{user_input}")
        conversation_history = truncate_conversation_history(conversation_history)
        prompt = '\n'.join(conversation_history)
        response = call_agent_app(prompt)
        if isinstance(response, str) and (response.startswith("错误") or response.startswith("调用失败")):
            print(f"助手：{response}")
            continue
        conversation_history.append(f"助手：{response}")
        print("助手：", end="", flush=True)
        stream_output(response)


if __name__ == "__main__":
    main()
