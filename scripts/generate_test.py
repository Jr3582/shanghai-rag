import os
import json
import anthropic
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
all_chunk_vectors = model.encode(texts)
cache = {}
def run_pipeline(query):
    if query in cache:
        return cache[query]

    if len(query) > 300:
        return "Question too long, please shorten it."
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)
    scored_chunks = list(enumerate(scores))

    top_3 = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:3]

    query_vector = model.encode(query)
    dense_scores = util.cos_sim(query_vector, all_chunk_vectors)[0]
    normalize_dense_scores = normalize(dense_scores)
    normalize_scores = normalize(scores)
    hybrid_score = []
    for d, b in zip(normalize_dense_scores, normalize_scores):
        hybrid_score.append(0.5 * d + 0.5 * b)
    scored_hybrid = list(enumerate(hybrid_score))
    top_20_hybrid = sorted(scored_hybrid, key=lambda x: x[1], reverse=True)[:20]

    pairs = []
    for index, chunk in top_20_hybrid:
        pairs.append([query, texts[index]])

    scores = reranker.predict(pairs)
    scored_rerank = list(enumerate(scores))
    top_3_rerank = sorted(scored_rerank, key=lambda x: x[1], reverse=True)[:3]

    chunk_texts = []
    for rank_index, score in top_3_rerank:
        original_index = top_20_hybrid[rank_index][0]
        chunk_texts.append(texts[original_index])

    context = "\n\n".join(chunk_texts)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know."

    Context:
    {context}

    Question: {query}"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=250,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.content[0].text

    cache[query] = (chunk_texts, context, answer)
    return cache[query]