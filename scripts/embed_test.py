from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "The Battle of Shanghai began in August 1937."
sentence2 = "Japanese troops invaded Shanghai in 1937."
sentence3 = "The weather is nice today"
vectors = model.encode([sentence1, sentence2, sentence3])

similarity = util.cos_sim(vectors[0], vectors[1])
similarity2 = util.cos_sim(vectors[0], vectors[2])

print(type(vectors))
print(vectors.shape)
print(vectors[0][:5])  # just peek at the first 5 numbers
print(f"SIMILARITY 1: {similarity}")
print(f"SIMILARITY 2: {similarity2}")