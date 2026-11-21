# 🌲 Pinecone & Groq RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **Pinecone Vector Database**, and **Groq (Meta Llama 3.3 70B)**.

This application allows you to ingest documents, automatically chunk and index them into Pinecone using integrated dense embeddings, and query them with semantic search to generate accurate, context-grounded answers via Groq.

---

## 🚀 Features

- **Integrated Vector Embeddings:** Leverages Pinecone's serverless integrated embedding model (`llama-text-embed-v2` · 1024 dimensions) — no external or local embedding pipelines required.
- **Ultra-Fast LLM Inference:** Powered by Groq's high-speed LPU running Meta's `llama-3.3-70b-versatile`.
- **Intelligent Text Chunking:** Sliding window chunking with configurable size and overlap to preserve semantic continuity across chunk boundaries.
- **Layered Architecture:** Clean separation of concerns with a modular `src/` layout (Schemas, Services, API Routers).
- **Graceful Error Handling:** Comprehensive `try-except` blocks returning meaningful HTTP status codes (`400`, `500`) with actionable error details.
- **Interactive Documentation:** Automatic Swagger/OpenAPI documentation via FastAPI.

---

## 🧠 Architecture & RAG Pipeline

```
[1. Document Ingestion]
      │
      ▼
┌──────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  Raw Text    │ ───► │  chunk_text()           │ ───► │  Pinecone Index      │
│  (Title/Body)│      │  (500 chars / 50 ovlp)  │      │  (Integrated Embed)  │
└──────────────┘      └─────────────────────────┘      └──────────────────────┘
                                                                  ▲
[2. Query & Generation]                                           │
                                                                  │ Semantic Search
┌──────────────┐      ┌─────────────────────────┐                 │ (Top 3 Chunks)
│ User Query   │ ───► │ Pinecone EmbedQuery     │ ────────────────┘
└──────────────┘      └─────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐      ┌──────────────────────┐
                      │ Context Assembly        │ ───► │ Groq LLM             │ ───► Final Answer
                      │ (Retrieved Chunks)      │      │ (Llama 3.3 70B)      │
                      └─────────────────────────┘      └──────────────────────┘
```

---

## 📁 Project Structure

```
Pinecone Learn/
├── .env                  # Environment secrets & credentials
├── pyproject.toml        # Project dependencies and configuration
├── uv.lock               # Deterministic dependency lockfile
├── README.md             # Project documentation
├── main.py               # Application entrypoint & FastAPI initialization
└── src/
    ├── __init__.py       # Package marker
    ├── api.py            # FastAPI Router and endpoint handlers (/upload-document, /ask)
    ├── schemas.py        # Pydantic request and response schemas
    └── services.py       # Core RAG engine: chunking, Pinecone upsert & Groq generation
```

---

## 🛠️ Prerequisites & Installation

### 1. Prerequisites
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or standard `pip`

### 2. Clone and Setup Environment

```bash
# Clone or navigate to the project directory
cd "Pinecone Learn"

# Install dependencies using uv
uv sync
```

Or install using standard pip:
```bash
pip install -r <(uv pip compile pyproject.toml)
```

---

## ⚙️ Configuration (`.env`)

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-learn
GROQ_API_KEY=your_groq_api_key_here
```

### Where to get your API Keys:
1. **Pinecone API Key:**
   - Sign up at [Pinecone Console](https://app.pinecone.io/).
   - Go to **API Keys** and copy your default key.
2. **Groq API Key:**
   - Sign up for free at [Groq Console](https://console.groq.com/keys).
   - Click **Create API Key** and copy your `gsk_...` key.

---

## 🗄️ Pinecone Index Setup

When creating your index in the [Pinecone Console](https://app.pinecone.io/):

| Setting | Value |
| :--- | :--- |
| **Index Name** | `rag-learn` |
| **Capacity** | `Serverless` |
| **Cloud / Region** | `AWS` / `us-east-1` (Free Starter) |
| **Schema: `chunk_text`** | Type: `String`, Configuration: `Dense embedding` (`llama-text-embed-v2` · 1024 dim · cosine) |
| **Schema: `title`** | Type: `String`, Configuration: `Full-text search` |

---

## 🚦 Running the Application

Start the local development server with hot-reload:

```bash
uv run fastapi dev main.py
```

The server will start at:
```
http://127.0.0.1:8000
```

Access the interactive API documentation (Swagger UI) at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 📖 API Reference & Usage

### 1. Health Check
* **Endpoint:** `GET /`
* **Response:**
  ```json
  {
    "message": "Welcome to Pinecone RAG API!"
  }
  ```

---

### 2. Upload and Index Document
* **Endpoint:** `POST /upload-document`
* **Request Body:**
  ```json
  {
    "title": "NovaTech Company Policy 2026",
    "content": "NovaTech Solutions was founded in 2024 by Dr. Ayman Rahman. The company headquarters is located in Banani, Dhaka, Bangladesh. NovaTech builds advanced AI systems, and their flagship product is called 'AuraScan 3.0', which detects faults in solar power plants using drone imagery. The company follows a hybrid work policy where all engineers can work from home every Wednesday and Friday. NovaTech provides a free buffet lunch at the office every Tuesday and Thursday. Every full-time employee receives an annual learning and development budget of 1200 USD to buy books, courses, or attend tech conferences."
  }
  ```
* **Response:**
  ```json
  {
    "message": "Successfully uploaded and indexed 2 chunks!",
    "chunks_indexed": 2
  }
  ```
* **cURL Example:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/upload-document" \
       -H "Content-Type: application/json" \
       -d '{
         "title": "NovaTech Company Policy 2026",
         "content": "NovaTech Solutions was founded in 2024 by Dr. Ayman Rahman..."
       }'
  ```

---

### 3. Ask a Question (RAG Query)
* **Endpoint:** `POST /ask`
* **Request Body:**
  ```json
  {
    "query": "Which days can employees work from home and what is the learning budget?"
  }
  ```
* **Response:**
  ```json
  {
    "question": "Which days can employees work from home and what is the learning budget?",
    "answer": "According to the company policy, employees can work from home every Wednesday and Friday. Additionally, each full-time employee receives an annual learning and development budget of 1,200 USD.",
    "sources": [
      "The company follows a hybrid work policy where all engineers can work from home every Wednesday and Friday. NovaTech provides a free buffet lunch at the office every Tuesday and Thursday. Every full-time employee receives an annual learning and development budget of 1200 USD to buy books, courses, or attend tech conferences."
    ]
  }
  ```
* **cURL Example:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/ask" \
       -H "Content-Type: application/json" \
       -d '{"query": "Which days can employees work from home?"}'
  ```

---

## 🛡️ Error Handling

The application handles common edge cases gracefully:
- **Empty payloads:** Returns `400 Bad Request` with descriptive feedback (`"Content cannot be empty."`).
- **External API or Network Failures:** Catches Pinecone/Groq connection errors and returns `500 Internal Server Error` with the root cause instead of crashing the server process.
- **Out-of-domain Questions:** If a question cannot be answered from the uploaded context, the LLM politely responds with:
  > *"Sorry, no relevant information was found for your question."*

---

## 📜 License

This project is open-source and available under the **MIT License**.
