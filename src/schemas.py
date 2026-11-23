from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """Schema for document upload request payload."""
    title: str = Field(..., description="Title of the document to index", example="NovaTech Company Policy 2026")
    content: str = Field(..., description="Full text content of the document", example="NovaTech was founded in 2024...")


class QueryRequest(BaseModel):
    """Schema for asking a standalone question against indexed documents."""
    query: str = Field(..., description="User query or question", example="What is the work from home policy?")


class ChatMessage(BaseModel):
    """Schema representing a single message in conversation history."""
    role: str = Field(..., description="Message author role: 'user' or 'assistant'", example="user")
    content: str = Field(..., description="Text content of the message", example="What is the leave policy?")


class ConversationalQueryRequest(BaseModel):
    """Schema for conversational queries supporting chat history and streaming."""
    query: str = Field(..., description="Current question asked by the user", example="Can I take it in December?")
    history: list[ChatMessage] = Field(
        default=[],
        description="Chronological list of prior messages in the conversation",
    )
    use_hybrid: bool = Field(
        default=True,
        description="Whether to use Hybrid Search (Dense + BM25 + Reranker) or basic vector search",
    )
