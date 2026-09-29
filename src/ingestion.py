import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def ingest_documents():
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Add explicit metadata required by the assignment.
    for chunk in chunks:
        chunk.metadata["source"] = PDF_PATH
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["text"] = chunk.page_content

    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings and uploading to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()