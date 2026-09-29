import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


class GraphState(TypedDict):
    question: str
    context: list[str]
    answer: str
    confidence_score: float


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def retrieve(state: GraphState):
    question = state["question"]

    results = vector_store.similarity_search_with_score(
        question,
        k=12
    )

    # For challenge/limitation questions, perform an additional
    # targeted retrieval so relevant challenge sections are not missed.
    question_lower = question.lower()

    if "challenge" in question_lower or "limitation" in question_lower:
        targeted_results = vector_store.similarity_search_with_score(
            "challenges limitations risks of Agentic AI systems",
            k=10
        )

        results.extend(targeted_results)

    # Remove duplicate chunks while preserving retrieval order.
    unique_chunks = []
    seen = set()

    for doc, score in results:
        chunk_text = doc.page_content

        if chunk_text not in seen:
            seen.add(chunk_text)
            unique_chunks.append((doc, score))

    unique_chunks.sort(key=lambda item: item[1], reverse=True)

    context = [doc.page_content for doc, score in unique_chunks[:10]]

    if unique_chunks:
        top_score = float(unique_chunks[0][1])
        confidence_score = max(0.0, min(1.0, top_score))
    else:
        confidence_score = 0.0

    return {
        "context": context,
        "confidence_score": confidence_score
    }

def generate(state: GraphState):
    question = state["question"]
    context = state["context"]

    context_text = "\n\n---\n\n".join(context)

    prompt = f"""
You are a strict document-based question answering assistant.

Answer the user's question ONLY using the information provided in the
context below.

If the context does not contain enough information to answer the question,
say exactly:

"I cannot answer this question because the provided eBook context does not contain the required information."

Use only facts that are explicitly supported by the context.
Do not combine unrelated statements from different parts of the context
to create a new claim.
Do not interpret descriptions of non-agentic AI systems as limitations
of Agentic AI unless the context explicitly makes that connection.
Do not use outside knowledge.
Do not guess.
Do not invent facts.

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def grade_groundedness(state: GraphState):
    answer = state["answer"].lower()
    context = " ".join(state["context"]).lower()

    if not state["context"]:
        return {"confidence_score": 0.0}

    # If the model explicitly refused because the information
    # was not found in the eBook, return a low confidence score.
    refusal = "cannot answer this question because" in answer

    if refusal:
        return {"confidence_score": 0.0}

    answer_words = {
        word.strip(".,!?():;\"'")
        for word in answer.split()
        if len(word.strip(".,!?():;\"'")) > 3
    }

    context_words = {
        word.strip(".,!?():;\"'")
        for word in context.split()
        if len(word.strip(".,!?():;\"'")) > 3
    }

    if not answer_words:
        return {"confidence_score": 0.0}

    overlap = len(answer_words & context_words) / len(answer_words)

    score = round(min(1.0, overlap), 2)

    return {
        "confidence_score": score
    }
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_groundedness", grade_groundedness)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_groundedness")
workflow.add_edge("grade_groundedness", END)

rag_graph = workflow.compile()


def ask_question(question: str):
    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "confidence_score": 0.0
    })

    return result