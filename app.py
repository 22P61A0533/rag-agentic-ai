from fastapi import FastAPI
from src.graph import ask_question

app = FastAPI(title="Agentic AI RAG Chatbot")


@app.get("/")
def root():
    return {"message": "Agentic AI RAG Chatbot is running"}


@app.get("/ask")
def ask(question: str):
    result = ask_question(question)

    return {
        "query": question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["confidence_score"]
    }