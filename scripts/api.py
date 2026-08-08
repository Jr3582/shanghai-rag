from fastapi import FastAPI
from pydantic import BaseModel
from generate_test import run_pipeline
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

class Question(BaseModel):
    query: str

@app.post("/ask")
@limiter.limit("5/minute")
def ask_question(request: Request, q: Question):
    chunk_texts, context, answer = run_pipeline(q.query)
    return {"answer": answer}