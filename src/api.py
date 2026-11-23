from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from src.schemas import (
    DocumentUploadRequest,
    QueryRequest,
    ConversationalQueryRequest,
)
from src import services

# Initialize API router instance for grouping endpoints
router = APIRouter()


@router.get("/")
def welcome():
    """Health check endpoint to verify that the API is up and running."""
    return {"message": "Welcome to Pinecone & Groq RAG API!"}


@router.post("/upload-document")
def upload_document(req: DocumentUploadRequest):
    """
    Upload and index raw text content into the Pinecone vector database.
    """
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    try:
        indexed_count = services.upload_and_index_document(req.title, req.content)
        if indexed_count == 0:
            raise HTTPException(status_code=400, detail="No valid content chunks could be created.")

        return {
            "message": f"Successfully uploaded and indexed {indexed_count} chunks!",
            "chunks_indexed": indexed_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document to Pinecone: {str(e)}",
        )


@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document, extract text automatically, and index it into Pinecone.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        chunks_indexed, char_count = services.upload_and_index_file(
            filename=file.filename,
            content_bytes=content_bytes,
        )

        return {
            "filename": file.filename,
            "characters_extracted": char_count,
            "chunks_indexed": chunks_indexed,
            "message": f"Successfully extracted and indexed {chunks_indexed} chunks from '{file.filename}'!",
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file '{file.filename}': {str(e)}",
        )


@router.post("/ask")
def ask_question(req: QueryRequest):
    """
    Standard RAG query endpoint using single-pass Dense Vector Search.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = services.answer_query(req.query)
        return {
            "question": req.query,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your question: {str(e)}",
        )


@router.post("/ask-advanced")
def ask_question_advanced(req: QueryRequest):
    """
    Advanced RAG query endpoint using:
    - Hybrid Search (Dense Embeddings + Sparse BM25)
    - Reciprocal Rank Fusion (RRF)
    - Cross-Encoder Reranking (bge-reranker-v2-m3)
    - Groq Llama 3.3 70B Generation
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = services.answer_query_hybrid(req.query)
        return {
            "question": req.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "pipeline_type": result.get("pipeline_type"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your advanced question: {str(e)}",
        )


@router.post("/chat-stream")
def chat_stream(req: ConversationalQueryRequest):
    """
    Conversational RAG endpoint with:
    - Chat history / multi-turn context memory
    - Query condensation (follow-up question rephrasing)
    - Hybrid Search + Reranking
    - Real-time token streaming (ChatGPT-style word-by-word response)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Convert Pydantic history objects to plain dicts
        history_dicts = [msg.model_dump() for msg in req.history]

        token_generator = services.stream_conversational_rag(
            query=req.query,
            history=history_dicts,
            use_hybrid=req.use_hybrid,
        )

        return StreamingResponse(
            token_generator,
            media_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream conversational response: {str(e)}",
        )
