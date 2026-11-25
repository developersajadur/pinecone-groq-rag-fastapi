import os
import uuid
import io
from collections.abc import Generator
from dotenv import load_dotenv
from pinecone import Pinecone, EmbedQuery, TextQuery
from groq import Groq
from pypdf import PdfReader
from src.cache import rag_cache

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

# Model ID for Groq inference (configurable via .env, defaults to active Groq model)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text string into smaller overlapping chunks.

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


def extract_text_from_file_bytes(filename: str, content_bytes: bytes) -> str:
    """
    Extract readable text from uploaded file bytes (supports .pdf and .txt).

    Args:
        filename (str): Name of the file with extension.
        content_bytes (bytes): Raw bytes content of the uploaded file.

    Returns:
        str: Extracted text from the file.

    Raises:
        ValueError: If file format is unsupported or no text could be extracted.
    """
    lower_name = filename.lower()
    extracted_text = ""

    if lower_name.endswith(".pdf"):
        try:
            pdf_stream = io.BytesIO(content_bytes)
            reader = PdfReader(pdf_stream)
            pages_text = []
            for page_idx, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                if page_content.strip():
                    pages_text.append(f"--- Page {page_idx + 1} ---\n{page_content.strip()}")
            extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {str(e)}")

    elif lower_name.endswith(".txt") or lower_name.endswith(".md"):
        extracted_text = content_bytes.decode("utf-8", errors="ignore").strip()

    else:
        raise ValueError("Unsupported file format. Please upload a .pdf, .txt, or .md file.")

    if not extracted_text.strip():
        raise ValueError("The uploaded file does not contain any readable text.")

    return extracted_text.strip()


def upload_and_index_document(title: str, content: str) -> int:
    """
    Process, chunk, and upsert a document into the Pinecone vector database.

    Args:
        title (str): The title or filename of the document.
        content (str): The body text of the document.

    Returns:
        int: The number of chunks successfully indexed in Pinecone.
    """
    chunks = chunk_text(content)
    if not chunks:
        return 0

    documents_to_upsert = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{title.replace(' ', '_').lower()}_{i}_{uuid.uuid4().hex[:6]}"
        documents_to_upsert.append({
            "_id": doc_id,
            "title": title,
            "chunk_text": chunk,
        })

    index.documents.upsert(
        namespace="default",
        documents=documents_to_upsert,
    )
    # Invalidate stale cache entries whenever new knowledge is added
    rag_cache.clear()
    return len(documents_to_upsert)


def upload_and_index_file(filename: str, content_bytes: bytes) -> tuple[int, int]:
    """
    Extracts text from file bytes and indexes it into Pinecone.

    Args:
        filename (str): Name of the file.
        content_bytes (bytes): Raw bytes of the uploaded file.

    Returns:
        tuple[int, int]: (chunks_indexed, character_count)
    """
    text = extract_text_from_file_bytes(filename, content_bytes)
    chunks_count = upload_and_index_document(title=filename, content=text)
    return chunks_count, len(text)


def reciprocal_rank_fusion(
    dense_results: list,
    keyword_results: list,
    k: int = 60,
) -> list[str]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine Dense and BM25 results.
    Formula: RRF_Score = 1 / (k + rank)
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, str] = {}

    for rank, doc in enumerate(dense_results):
        chunk = doc.get("chunk_text")
        if chunk:
            doc_id = doc.id
            chunk_map[doc_id] = chunk
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc in enumerate(keyword_results):
        chunk = doc.get("chunk_text")
        if chunk:
            doc_id = doc.id
            chunk_map[doc_id] = chunk
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
    return [chunk_map[did] for did in sorted_doc_ids]


def answer_query(query: str) -> dict:
    """Standard Dense Vector Search RAG with Redis Cache."""
    cached = rag_cache.get(query)
    if cached:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "cache_hit": True,
            "cache_type": cached["cache_type"],
        }

    search_result = index.documents.search(
        namespace="default",
        top_k=3,
        score_by=[EmbedQuery(query=query, field="chunk_text")],
        include_fields=["title", "chunk_text"],
    )

    retrieved_chunks = [
        match.get("chunk_text")
        for match in search_result.matches
        if match.get("chunk_text")
    ]

    if not retrieved_chunks:
        return {
            "answer": "Sorry, no relevant information was found in the indexed documents.",
            "sources": [],
        }

    context_text = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question strictly based on the following context.
If the answer cannot be found in the context, reply: "Sorry, no relevant information was found for your question."

Context:
{context_text}

Question:
{query}

Answer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
    )

    answer_text = chat_completion.choices[0].message.content
    rag_cache.set(query=query, answer=answer_text, sources=retrieved_chunks)

    return {
        "answer": answer_text,
        "sources": retrieved_chunks,
    }


def answer_query_hybrid(query: str, top_k_retrieval: int = 10, final_top_n: int = 3) -> dict:
    """Advanced Hybrid (Dense + BM25) + RRF + Cross-Encoder Reranking RAG with Redis Cache."""
    cached = rag_cache.get(query)
    if cached:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "pipeline_type": f"cache_hit ({cached['cache_type']}, sim={cached['similarity']})",
        }

    dense_result = index.documents.search(
        namespace="default",
        top_k=top_k_retrieval,
        score_by=[EmbedQuery(query=query, field="chunk_text")],
        include_fields=["title", "chunk_text"],
    )

    keyword_result = index.documents.search(
        namespace="default",
        top_k=top_k_retrieval,
        score_by=[TextQuery(query=query, fields=["chunk_text"])],
        include_fields=["title", "chunk_text"],
    )

    merged_candidates = reciprocal_rank_fusion(
        dense_results=dense_result.matches,
        keyword_results=keyword_result.matches,
    )

    if not merged_candidates:
        return {
            "answer": "Sorry, no relevant information was found in the indexed documents.",
            "sources": [],
            "pipeline_type": "hybrid_reranking",
        }

    try:
        rerank_response = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=merged_candidates[:15],
            top_n=final_top_n,
            return_documents=True,
        )
        final_chunks = [item.document["text"] for item in rerank_response.data]
    except Exception:
        final_chunks = merged_candidates[:final_top_n]

    context_text = "\n\n---\n\n".join(final_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question strictly based on the following context.
If the answer cannot be found in the context, reply: "Sorry, no relevant information was found for your question."

Context:
{context_text}

Question:
{query}

Answer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
    )

    answer_text = chat_completion.choices[0].message.content
    rag_cache.set(query=query, answer=answer_text, sources=final_chunks)

    return {
        "answer": answer_text,
        "sources": final_chunks,
        "pipeline_type": "hybrid_rrf_reranking",
    }


def condense_conversational_query(query: str, history: list[dict]) -> str:
    """
    Contextualize a follow-up user query using conversation history.
    Example:
    History: "What is NovaTech?" -> "An AI company."
    User asks: "Where is it located?"
    Condensed: "Where is NovaTech located?"
    """
    if not history:
        return query

    # Format the recent history turns
    recent_history = history[-4:]
    history_text = "\n".join(
        f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
        for msg in recent_history
    )

    condensation_prompt = f"""Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone, self-contained search query.
Do NOT answer the question. Only return the rewritten search query. If the question is already self-contained, return it unchanged.

Conversation History:
{history_text}

Follow-up Question: {query}
Standalone Search Query:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": condensation_prompt}],
            model=GROQ_MODEL,
            max_tokens=60,
            temperature=0.0,
        )
        condensed = response.choices[0].message.content.strip().strip('"')
        return condensed or query
    except Exception:
        return query


def summarize_older_history(older_messages: list[dict]) -> str:
    """
    Summarize conversation turns older than the recent 6-message window.
    Preserves key user details (e.g., user's name, preferences) and main topics
    while keeping context window token usage compact.
    """
    if not older_messages:
        return ""

    transcript = "\n".join(
        f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
        for msg in older_messages
    )

    prompt = (
        "Summarize the following earlier conversation concisely in 2-4 sentences. "
        "Be sure to preserve important facts like the user's name, preferences, and key topics.\n\n"
        f"Earlier Conversation:\n{transcript}\n\nSummary:"
    )

    try:
        res = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            max_tokens=150,
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return ""


def stream_conversational_rag(
    query: str,
    history: list[dict],
    use_hybrid: bool = True,
) -> Generator[str, None, None]:
    """
    Full conversational RAG pipeline with Sliding Window + Summary Memory:
    1. Rephrases follow-up questions using conversation history into a standalone search query.
    2. Checks Redis Semantic Cache only for standalone queries (when history is empty).
    3. Retrieves the most relevant chunks using Hybrid Search or Dense Vector Search.
    4. Keeps the last 6 messages verbatim and summarizes any older turns beyond 6.
    5. Streams response tokens directly from Groq LLM.
    """
    # Step 1: Contextualize the search query
    search_query = condense_conversational_query(query, history)

    # Step 1.5: Check Redis Semantic Cache ONLY when there is no prior conversation history
    if not history:
        cached = rag_cache.get(search_query)
        if cached:
            yield cached["answer"]
            return

    # Step 2: Retrieve relevant chunks
    if use_hybrid:
        # Use hybrid retrieval (Dense + BM25) + Reranking
        dense_result = index.documents.search(
            namespace="default",
            top_k=8,
            score_by=[EmbedQuery(query=search_query, field="chunk_text")],
            include_fields=["title", "chunk_text"],
        )
        keyword_result = index.documents.search(
            namespace="default",
            top_k=8,
            score_by=[TextQuery(query=search_query, fields=["chunk_text"])],
            include_fields=["title", "chunk_text"],
        )
        candidates = reciprocal_rank_fusion(dense_result.matches, keyword_result.matches)
        try:
            if candidates:
                rerank_res = pc.inference.rerank(
                    model="bge-reranker-v2-m3",
                    query=search_query,
                    documents=candidates[:10],
                    top_n=3,
                    return_documents=True,
                )
                retrieved_chunks = [item.document["text"] for item in rerank_res.data]
            else:
                retrieved_chunks = []
        except Exception:
            retrieved_chunks = candidates[:3]
    else:
        # Simple Dense Vector Search
        search_res = index.documents.search(
            namespace="default",
            top_k=3,
            score_by=[EmbedQuery(query=search_query, field="chunk_text")],
            include_fields=["title", "chunk_text"],
        )
        retrieved_chunks = [
            m.get("chunk_text") for m in search_res.matches if m.get("chunk_text")
        ]

    # Step 3: Handle empty context
    if not retrieved_chunks and not history:
        yield "Sorry, no relevant information was found in the indexed documents for your question."
        return

    context_text = (
        "\n\n---\n\n".join(retrieved_chunks)
        if retrieved_chunks
        else "No matching document chunks were found for this query."
    )

    # Step 3.5: Summarize older messages if history exceeds the 6-message window
    history_summary = ""
    if len(history) > 6:
        history_summary = summarize_older_history(history[:-6])

    recent_messages = history[-6:]
    summary_section = (
        f"\n\nEarlier Conversation Summary:\n{history_summary}"
        if history_summary
        else ""
    )

    # Step 4: Build conversational messages payload
    system_instruction = (
        "You are a helpful, accurate AI assistant. "
        "Answer the user's question using the provided Document Context AND the Conversation History "
        "(including any personal details like the user's name or preferences mentioned in the chat). "
        "If the question asks about a document fact that is not in the Document Context or Conversation History, reply: "
        "'Sorry, no relevant information was found for your question.'\n\n"
        f"Document Context:\n{context_text}"
        f"{summary_section}"
    )

    messages = [{"role": "system", "content": system_instruction}]

    # Include up to the most recent 6 conversation messages verbatim
    for msg in recent_messages:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })

    # Add current user query
    messages.append({"role": "user", "content": query})

    # Step 5: Stream tokens from Groq and cache standalone responses
    stream = groq_client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
        stream=True,
    )

    collected_tokens: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            collected_tokens.append(delta)
            yield delta

    # Save complete answer to Redis cache only for standalone queries (no history)
    if collected_tokens and not history and retrieved_chunks:
        rag_cache.set(
            query=search_query,
            answer="".join(collected_tokens),
            sources=retrieved_chunks,
        )
