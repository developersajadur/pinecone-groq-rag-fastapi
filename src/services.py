import os
import uuid
from dotenv import load_dotenv
from pinecone import Pinecone, EmbedQuery
from groq import Groq

# Load environment variables from .env file
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Client Initializations
# ---------------------------------------------------------------------------
# Initialize Pinecone client with the API key from environment variables
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"].strip())

# Connect to the target Pinecone vector index
index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-learn").strip()
index = pc.Index(index_name)

# Initialize Groq client for fast LLM inference
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text string into smaller overlapping chunks.

    Why chunking is needed:
    Large documents need to be broken into smaller segments so that vector
    embeddings capture specific concepts effectively and fit within LLM context
    windows. Adding overlap prevents critical context from being split across boundaries.

    Args:
        text (str): The raw text content of the document.
        chunk_size (int): Maximum number of characters in each chunk (default: 500).
        overlap (int): Number of overlapping characters between consecutive chunks (default: 50).

    Returns:
        list[str]: A list of text chunk strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Slide forward by (chunk_size - overlap) to maintain text continuity
        start += chunk_size - overlap
    return chunks


def upload_and_index_document(title: str, content: str) -> int:
    """
    Process, chunk, and upsert a document into the Pinecone vector database.

    Workflow:
    1. Chunks the input content using `chunk_text`.
    2. Generates a unique document ID for each chunk.
    3. Builds the record dictionary containing metadata (title and chunk_text).
    4. Upserts documents into Pinecone (Pinecone automatically converts
       `chunk_text` into dense vector embeddings using its integrated model).

    Args:
        title (str): The title of the document.
        content (str): The body text of the document.

    Returns:
        int: The number of chunks successfully indexed in Pinecone.
    """
    chunks = chunk_text(content)
    if not chunks:
        return 0

    documents_to_upsert = []
    for i, chunk in enumerate(chunks):
        # Generate a unique ASCII identifier for each chunk: title_index_uuid
        doc_id = f"{title.replace(' ', '_').lower()}_{i}_{uuid.uuid4().hex[:6]}"
        documents_to_upsert.append({
            "_id": doc_id,
            "title": title,
            "chunk_text": chunk,
        })

    # Upsert documents into the 'default' namespace
    index.documents.upsert(
        namespace="default",
        documents=documents_to_upsert,
    )
    return len(documents_to_upsert)


def answer_query(query: str) -> dict:
    """
    Retrieve relevant document chunks from Pinecone and generate an answer using Groq LLM.

    Workflow (RAG Pipeline):
    1. Semantic Search (Retrieval):
       - Converts the user's query into an embedding via Pinecone's `EmbedQuery`.
       - Retrieves the top 3 most semantically similar chunks from Pinecone.
    2. Fallback Check:
       - If no relevant chunks match, returns an informative message.
    3. Context Assembly:
       - Combines retrieved chunks into a single formatted context string.
    4. Generation (LLM Answer):
       - Sends the context + user question to Groq (Llama 3.3 70B).
       - The LLM is instructed to answer strictly based on the provided context.

    Args:
        query (str): The user's question or search query.

    Returns:
        dict: A dictionary containing:
            - 'answer': The LLM-generated answer.
            - 'sources': List of context chunks used to formulate the answer.
    """
    # Step 1: Perform semantic similarity search in Pinecone
    search_result = index.documents.search(
        namespace="default",
        top_k=3,
        score_by=[EmbedQuery(query=query, field="chunk_text")],
        include_fields=["title", "chunk_text"],
    )

    # Step 2: Extract text chunks from the top matches
    retrieved_chunks = [
        match.get("chunk_text")
        for match in search_result.matches
        if match.get("chunk_text")
    ]

    # Return early if no relevant documents were found
    if not retrieved_chunks:
        return {
            "answer": "Sorry, no relevant information was found in the indexed documents.",
            "sources": [],
        }

    # Step 3: Combine retrieved chunks into a unified context
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    # Step 4: Construct the system-guided prompt for the LLM
    prompt = f"""You are a helpful assistant. Answer the user's question strictly based on the following context.
If the answer cannot be found in the context, reply: "Sorry, no relevant information was found for your question."

Context:
{context_text}

Question:
{query}

Answer:"""

    # Step 5: Generate answer using Groq (Llama 3.3 70B Versatile)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )

    return {
        "answer": chat_completion.choices[0].message.content,
        "sources": retrieved_chunks,
    }
