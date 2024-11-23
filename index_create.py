import faiss
import numpy as np
import pandas as pd

df = pd.read_csv("data_with_embeddings.csv")
assert "embedding" in df.columns, "请检查是否正确"


vectors = np.array([eval(embedding) for embedding in df["embedding"]]).astype("float32")

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors) # type: ignore

index_file = "vector_index.index"
faiss.write_index(index, index_file)
print(f"FAISS index saved to {index_file}")

loaded_index = faiss.read_index(index_file)
print(f"Number of vectors in loaded index: {loaded_index.ntotal}")

query_vector = vectors[0].reshape(1, -1)
D, I = loaded_index.search(query_vector, k=5)
print(f"Distances: {D}, Indices: {I}")
