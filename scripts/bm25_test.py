from rank_bm25 import BM25Okapi
import json
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized = []
    for s in scores:
        norm = (s - min_score) / (max_score - min_score)
        normalized.append(norm)
    return normalized

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

with open("../data/chunks.json", "r") as f:
    data = json.load(f)

texts = []

for chunk in data:
    texts.append(chunk["text"])

# tokenized_docs = list of lists, each inner list is one document's words
tokenized_docs = [text.split() for text in texts]

bm25 = BM25Okapi(tokenized_docs)

query = "Japanese troops invaded Shanghai"
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)
scored_chunks = list(enumerate(scores))

top_3 = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:3]

# for index, score in top_3:
#     print(index, score)
#     print(texts[index])
#     print("---")

query_vector = model.encode(query)
all_chunk_vectors = model.encode(texts)
dense_scores = util.cos_sim(query_vector, all_chunk_vectors)[0]
normalize_dense_scores = normalize(dense_scores)
normalize_scores = normalize(scores)
hybrid_score = []
for d, b in zip(normalize_dense_scores, normalize_scores):
    hybrid_score.append(0.5 * d + 0.5 * b)
scored_hybrid = list(enumerate(hybrid_score))
top_20_hybrid = sorted(scored_hybrid, key=lambda x: x[1], reverse=True)[:20]

for index in [409, 285, 380]:
    print(index, "dense:", normalize_dense_scores[index], "bm25:", normalize_scores[index])

for index, score in top_20_hybrid:
    print(index, score)
    print(texts[index])
    print("---")

pairs = []
for index, chunk in top_20_hybrid:
    pairs.append([query, texts[index]])

scores = reranker.predict(pairs)
scored_rerank = list(enumerate(scores))
top_3_rerank = sorted(scored_rerank, key=lambda x: x[1], reverse=True)[:3]

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for rank_index, score in top_3_rerank:
    original_index = top_20_hybrid[rank_index][0]
    print(original_index, score)
    print(texts[original_index])
    print("---")

