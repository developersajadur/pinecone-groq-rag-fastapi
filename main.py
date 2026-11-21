from fastapi import FastAPI
from src.api import router

# Initialize the main FastAPI application instance
app = FastAPI(
    title="Pinecone RAG Assistant",
    description="Production-ready RAG application using FastAPI, Pinecone Vector DB, and Groq LLM",
    version="1.0.0",
)

# Register the API routes defined in src/api.py
app.include_router(router)
