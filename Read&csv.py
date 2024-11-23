import pandas as pd
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
import dashscope

data_file = r"C:\Users\12393\Vscode\Python\AI\shoes.txt"
data = []

with open(data_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        data.append({"key": key, "value": value})

df = pd.DataFrame(data)
print("Total rows in DataFrame:", len(df))

dashscope.api_key = "sk-430eda3a5950475fa5dde7d12f51199f"

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

batch_size = 5
result_embeddings = []

for i in range(0, len(df), batch_size):
    batch = df["value"][i:i + batch_size].tolist()
    embeddings = embedder.get_text_embedding_batch(batch)
    result_embeddings.extend(embeddings)

df["embedding"] = result_embeddings

print("Total embeddings generated:", len(df["embedding"].dropna()))
print(df)

df.to_csv("data_with_embeddings.csv", index=False, encoding="utf-8")
