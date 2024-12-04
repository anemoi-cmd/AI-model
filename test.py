import requests
import json

# 定义API端点
url = "http://127.0.0.1:5000/chat"

# 定义要发送的消息
payload = {
    "message": "鞋子有些什么尺码？"
}

# 定义请求头
headers = {
    "Content-Type": "application/json"
}

# 发送POST请求
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 打印响应
print(response.json())
