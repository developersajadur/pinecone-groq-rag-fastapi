from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """Schema for document upload request payload."""
    title: str = Field(..., description="Title of the document to index", example="NovaTech Company Policy 2026")
    content: str = Field(..., description="Full text content of the document", example="NovaTech was founded in 2024...")


class QueryRequest(BaseModel):
    """Schema for asking a question against indexed documents."""
    query: str = Field(..., description="User query or question", example="What is the work from home policy?")
