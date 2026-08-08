import os
import json
import anthropic
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from generate_test import run_pipeline
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

judge_llm = LangchainLLMWrapper(
    ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
)

with open("../data/golden_dataset.json", "r") as f:
    golden = json.load(f)

questions = []
contexts = []
answers = []
ground_truths = []



for item in golden:
    chunk_texts, context, answer = run_pipeline(item["question"])
    questions.append(item["question"])
    contexts.append(chunk_texts)
    answers.append(answer)
    ground_truths.append(item["ground_truth"])

data = {
    "question": questions,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": ground_truths,
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_correctness, context_precision],
    llm=judge_llm,
    embeddings=judge_embeddings
)
print(results)