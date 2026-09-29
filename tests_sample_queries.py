from src.graph import ask_question


questions = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


for i, question in enumerate(questions, 1):
    print(f"\n{'=' * 70}")
    print(f"QUESTION {i}: {question}")
    print(f"{'=' * 70}")

    result = ask_question(question)

    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nCONFIDENCE SCORE:")
    print(result["confidence_score"])

    print("\nRETRIEVED CONTEXT CHUNKS:")
    print(len(result["context"]))