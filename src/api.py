from fastapi import APIRouter, HTTPException
from src.schemas import DocumentUploadRequest, QueryRequest
from src import services

# Initialize API router instance for grouping endpoints
router = APIRouter()


@router.get("/")
def welcome():
    """Health check endpoint to verify that the API is up and running."""
    return {"message": "Welcome to Pinecone RAG API!"}


@router.post("/upload-document")
def upload_document(req: DocumentUploadRequest):
    """
    Upload and index a document into the Pinecone vector database.

    - Validates that title and content are not empty.
    - Chunks the text and stores vectors with metadata in Pinecone.
    - Handles exceptions and returns appropriate HTTP status codes.
    """
    # Validate request payload
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    try:
        # Call the service layer to process and index the document
        indexed_count = services.upload_and_index_document(req.title, req.content)
        if indexed_count == 0:
            raise HTTPException(status_code=400, detail="No valid content chunks could be created.")

        return {
            "message": f"Successfully uploaded and indexed {indexed_count} chunks!",
            "chunks_indexed": indexed_count,
        }
    except HTTPException:
        # Re-raise known HTTP exceptions without modification
        raise
    except Exception as e:
        # Catch unexpected errors (e.g., Pinecone network issues, invalid credentials)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document to Pinecone: {str(e)}",
        )


@router.post("/ask")
def ask_question(req: QueryRequest):
    """
    Ask a question against indexed documents using RAG (Retrieval-Augmented Generation).

    - Validates that the query string is not empty.
    - Retrieves top matching chunks from Pinecone.
    - Prompts Groq LLM (Llama 3.3 70B) to generate a grounded answer.
    """
    # Validate request payload
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Call the RAG pipeline service
        result = services.answer_query(req.query)
        return {
            "question": req.query,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except HTTPException:
        # Re-raise known HTTP exceptions
        raise
    except Exception as e:
        # Catch unexpected errors (e.g., LLM rate limit, search failure)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your question: {str(e)}",
        )
