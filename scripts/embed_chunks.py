import json
from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="../data/chroma_db")
collection = client.get_or_create_collection(name="shanghai_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("../data/chunks.json", "r") as f:
    data = json.load(f)

texts = []
for chunk in data:
    texts.append(chunk["text"])

vectors = model.encode(texts)

ids = []
embeddings = []
documents = []
metadatas = []

for i, chunk in enumerate(data):
    ids.append(f"{chunk['source']}_{chunk['chunk_id']}")
    embeddings.append(vectors[i].tolist())
    documents.append(chunk["text"])
    metadatas.append({"source": chunk["source"], "chunk_id": chunk["chunk_id"]})


collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

question_vector = model.encode("How many soldiers defended Sihang Warehouse?")

results = collection.query(
    query_embeddings=[question_vector.tolist()],
    n_results=3
)