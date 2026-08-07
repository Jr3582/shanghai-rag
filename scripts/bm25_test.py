from rank_bm25 import BM25Okapi

# tokenized_docs = list of lists, each inner list is one document's words
tokenized_docs = [doc.split() for doc in your_documents]

bm25 = BM25Okapi(tokenized_docs)

query = "How many soldiers"
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)