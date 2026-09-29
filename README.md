# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph** and **Pinecone** to answer questions strictly from an Agentic AI eBook.

## Features

* PDF ingestion using PyPDF
* Recursive text chunking with overlap
* Vector embeddings and semantic search
* Pinecone vector database
* LangGraph-based RAG workflow
* Retrieval, answer generation, and groundedness grading nodes
* Strict document-grounded responses
* Refusal when information is not available in the eBook
* FastAPI REST API
* Swagger/OpenAPI documentation
* Retrieved context chunks included in API responses
* Confidence/groundedness score included in API responses

## Architecture

```text
Agentic AI eBook (PDF)
        |
        v
    PDF Loader
        |
        v
   Text Chunking
   1000 chars
   200 overlap
        |
        v
Local Embeddings
        |
        v
    Pinecone
        |
        v
    LangGraph
        |
        +--> Retrieve
        |
        +--> Generate
        |
        +--> Grade Groundedness
        |
        v
   Grounded Answer
        |
        v
     FastAPI
```

## Technology Stack

* Python
* LangChain
* LangGraph
* Pinecone
* Sentence Transformers
* Ollama
* FastAPI
* PyPDF
* Uvicorn

## Project Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── tests_sample_queries.py
```

## Models

### Embeddings

The current implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is a local embedding model that produces 384-dimensional vectors.

### LLM

The application uses Groq's hosted LLM API for answer generation.

The current model configured in `src/graph.py` is:

`openai/gpt-oss-20b`

The Groq API key is loaded securely from the `GROQ_API_KEY` environment variable.

### Note about OpenAI Embeddings

The original assignment specifies OpenAI Embeddings with Pinecone.

This implementation uses the local Sentence Transformers embedding model instead of OpenAI Embeddings so that the project can run without paid OpenAI API usage.

Pinecone is still used as the vector database, and the rest of the RAG architecture remains implemented with LangGraph, retrieval, grounded answer generation, and groundedness grading.

## Pinecone

The Pinecone index uses:

```text
Index name: agentic-ai-index
Dimension: 384
Metric: cosine
```

The 384-dimensional index matches the `all-MiniLM-L6-v2` embedding model.

Each stored document chunk includes metadata such as:

* source PDF
* page number
* source text chunk

## Setup

### 1. Clone the repository

```powershell
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install and run Ollama

Install Ollama and make sure it is running.

Check available models:

```powershell
ollama list
```

Pull the required model if necessary:

```powershell
ollama pull llama3.2:latest
```

### 5. Configure environment variables

Create a `.env` file in the project root.

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

Keep `.env` private. Do not commit API keys to GitHub.

## Ingestion

Place the Agentic AI eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

Run:

```powershell
python src/ingestion.py
```

The ingestion process:

1. Loads the PDF.
2. Splits the document into chunks.
3. Creates vector embeddings.
4. Stores the vectors in Pinecone.
5. Stores source, page number, and text metadata.

The current eBook produces approximately 119 chunks using a chunk size of 1000 characters and an overlap of 200 characters.

## Running the API

Start FastAPI with:

```powershell
uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### GET `/ask`

Example:

```text
http://127.0.0.1:8000/ask?question=What%20is%20Agentic%20AI?
```

The API returns:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "Retrieved chunk from the eBook...",
    "Another retrieved chunk from the eBook..."
  ],
  "confidence_score": 1.0
}
```

## LangGraph Workflow

The RAG workflow consists of three main nodes:

### 1. Retrieve

The user's question is converted into an embedding and used to retrieve relevant chunks from Pinecone.

The system normally retrieves the top relevant chunks. Challenge/limitation questions also use a targeted semantic retrieval query to improve retrieval of challenge-related sections.

### 2. Generate

The retrieved context is passed to the local Ollama model.

The model is instructed to:

* answer only from the retrieved context
* avoid outside knowledge
* avoid guessing
* avoid inventing facts
* refuse when the required information is not present

### 3. Grade Groundedness

The generated answer is checked against the retrieved context using a deterministic lexical-overlap heuristic.

The resulting value is returned as `confidence_score`.

**Note:** This score is a groundedness/support heuristic, not a calibrated probability.

## Sample Queries

The project includes the six sample queries from the assignment.

Run:

```powershell
python tests_sample_queries.py
```

The test covers:

1. Core definition of Agentic AI
2. Main architectural components
3. Real-world industry use cases
4. Difference between Agentic AI and traditional generative AI chatbots
5. Challenges or limitations
6. An unrelated question to verify document-grounded refusal

For questions that are outside the eBook's content, the system returns:

```text
I cannot answer this question because the provided eBook context does not contain the required information.
```

## Grounding and Hallucination Control

The chatbot is designed to remain grounded in the provided eBook.

The generation prompt explicitly instructs the model not to:

* use outside knowledge
* guess
* invent facts
* combine unrelated statements to create unsupported claims

The LangGraph workflow also includes a separate groundedness grading step.

## Security

Do not commit the following to GitHub:

* `.env`
* API keys
* private credentials
* virtual environment files

Use `.env.example` as a template for required environment variables.

## Assignment Alignment

The implementation covers the major assignment requirements:

* PDF ingestion: Yes
* Text chunking with overlap: Yes
* Vector embeddings: Yes
* Pinecone vector database: Yes
* LangGraph orchestration: Yes
* Retrieval node: Yes
* Generation node: Yes
* Groundedness grading: Yes
* Document-grounded answers: Yes
* Out-of-document refusal: Yes
* FastAPI API: Yes
* Required response fields: Yes

The only implementation substitution is the embedding/LLM layer: local Sentence Transformers and Ollama are used instead of paid OpenAI Embeddings and OpenAI generation so the project can run without paid API usage.
