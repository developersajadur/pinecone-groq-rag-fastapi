# 🌲 Pinecone & Groq RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, **Pinecone Vector Database**, and **Groq (Meta Llama 3.3 70B)**.

This application allows you to ingest documents, automatically chunk and index them into Pinecone using integrated dense embeddings, and query them with semantic search to generate accurate, context-grounded answers via Groq.

---

## 🚀 Features

- **Integrated Vector Embeddings:** Leverages Pinecone's serverless integrated embedding model (`llama-text-embed-v2` · 1024 dimensions) — no external or local embedding pipelines required.
- **Ultra-Fast LLM Inference:** Powered by Groq's high-speed LPU (configurable via `GROQ_MODEL`, e.g., `openai/gpt-oss-120b`).
- **Redis Exact & Semantic Caching:** Two-layer Redis cache using SHA-256 exact keys and local FastEmbed + NumPy cosine similarity (`>= 0.88`) with automatic TTL and terminal hit/miss logging.
- **Hybrid Search & Cross-Encoder Reranking:** Dense Semantic Search + Sparse BM25 Keyword Search combined with Reciprocal Rank Fusion (RRF) and refined with `bge-reranker-v2-m3`.
- **PDF & Document Ingestion:** Directly upload `.pdf`, `.txt`, or `.md` files; text is parsed and indexed automatically.
- **Sliding Window + Summary Memory:** Keeps the last 6 messages verbatim, automatically summarizes older turns beyond 6 to preserve user details (like name/preferences) without overflowing the context window.
- **Real-Time Token Streaming:** Live word-by-word streaming responses (ChatGPT-style typing effect).
- **Interactive Streamlit Web UI:** Beautiful Cyber Dark Blue frontend for document uploads, diagnostics, and live chatting.
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
├── docker-compose.yml    # Redis Docker service configuration
├── pyproject.toml        # Project dependencies and configuration
├── uv.lock               # Deterministic dependency lockfile
├── README.md             # Project documentation
├── app_ui.py             # Interactive Streamlit Web UI
├── main.py               # Application entrypoint & FastAPI initialization
└── src/
    ├── __init__.py       # Package marker
    ├── api.py            # FastAPI Router and endpoint handlers
    ├── cache.py          # Redis Exact + Semantic Cache (FastEmbed + NumPy)
    ├── schemas.py        # Pydantic request and response schemas
    └── services.py       # Core RAG engine: chunking, Pinecone, Groq & Caching
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
GROQ_MODEL=openai/gpt-oss-120b
REDIS_PORT=6391
REDIS_URL=redis://localhost:6391/0
```

### Start Redis Cache (Docker Compose)
```bash
docker compose up -d
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

## 🖥️ Interactive Web UI (Streamlit)

The application includes a fully customizable, production-ready **Streamlit web application** (`app_ui.py`) designed with a sleek **Cyber Dark Blue Theme** and packed with real-time AI chat capabilities.

### 🚀 Launch the Web UI

```bash
uv run streamlit run app_ui.py
```

---

### 🎨 UI Layout & Architecture

```
┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│               SIDEBAR                │                       MAIN CHAT                        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│  📂 Document Ingestion               │  ┌──────────────────────────────────────────────────┐  │
│  ┌────────────────────────────────┐  │  │  🌲 Hero Banner: AI Knowledge Assistant          │  │
│  │ [📁 File Upload] [✍️ Paste Text]│  │  │  Model, Pinecone status, and description badge   │  │
│  └────────────────────────────────┘  │  └──────────────────────────────────────────────────┘  │
│                                      │                                                        │
│  ⚙️ Search Configuration             │  🤖 Assistant Message (Electric Blue Accent)           │
│  [🔘 Hybrid Search & Reranking]     │  "Hello! I am your AI Knowledge Assistant..."          │
│                                      │                                                        │
│  ⚡ System Diagnostics Card          │  👤 User Message (Royal Navy Accent)                   │
│  • Index: rag-learn                  │  "What is the company leave policy?"                   │
│  • LLM Engine: Active Groq Model     │                                                        │
│  • Search Mode: Hybrid + Rerank      │  ⚡ Thinking Indicator (Pulsing Animation)             │
│  • Streaming: Enabled                │  "Searching knowledge base & formulating answer..."    │
│                                      │                                                        │
│  [🗑️ Clear Conversation History]    │  💬 Chat Input: "Ask a question about your docs..."    │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

### 🌟 Key UI Features & Engineering Highlights

#### 1. 🌓 Modern Cyber & Dark Blue Aesthetic
* **Custom CSS Theme:** Built with high-contrast, eye-friendly dark slate tones (`#0F172A`, `#1E293B`, `#334155`) and vibrant royal blue accents (`#2563EB`, `#3B82F6`, `#60A5FA`).
* **Hero Banner Card:** Premium gradient card (`#0F172A` ➔ `#1E3A8A` ➔ `#2563EB`) with a glassmorphism pill badge and subtle elevation shadows.
* **Color-Coded Message Cards:**
  * **User Bubbles:** Deep navy gradient (`#172554` ➔ `#1E3A8A`) with an electric blue border (`#60A5FA`).
  * **AI Assistant Bubbles:** Dark slate card (`#1E293B`) with a neon blue left border (`#3B82F6`).

#### 2. ⚡ Animated "Thinking..." State Indicator
* **Visual Progress Feedback:** While Pinecone performs dense vector search, BM25 keyword matching, and cross-encoder reranking, a glowing animated badge is rendered in the chat bubble:
  > `⚡ Thinking and searching knowledge base...`
* As soon as the first token arrives from the LLM, the indicator automatically vanishes to make way for the answer.

#### 3. ✍️ Real-Time Paced Typewriter Streaming
* **Human-Paced Delivery:** Ultra-fast LLMs like Groq generate 300–500+ tokens/second, which can make answers appear abruptly. The UI implements an **18ms rhythmic typewriter pacing delay** (`paced_stream_generator`), giving users a smooth, authentic ChatGPT/Claude streaming experience.

#### 4. 📂 Dual Document Ingestion System (Sidebar Tabs)
* **📁 Tab 1: File Upload:**
  * Supports direct drag-and-drop of `.pdf`, `.txt`, and `.md` files (up to 25MB).
  * Automatically parses multi-page PDFs using `pypdf`, breaks text into overlapping chunks, and upserts them to Pinecone with instant feedback.
* **✍️ Tab 2: Paste Raw Text:**
  * Document Title and Content text area for indexing raw text, internal policies, or articles without needing a physical file.

#### 5. 🧠 Sliding Window + Summary Buffer Memory
* Maintains session conversation history (`st.session_state.messages`) and contextualizes follow-up questions before searching Pinecone.
* **Sliding Window (Last 6 Messages):** Keeps the 6 most recent messages verbatim for exact short-term continuity.
* **Automatic Summarization:** Any conversation turns older than 6 messages are automatically summarized by the LLM (`summarize_older_history`), preserving personal details (like the user's name or preferences) while keeping context-window token usage compact.
* **Smart Cache Bypass:** Standalone queries use Redis cache, while follow-up conversational turns bypass cache so personal chat context is always respected.

#### 6. ⚙️ Interactive Controls & Diagnostics Card
* **Search Mode Toggle:** Switch seamlessly between **Hybrid Search + Reranking** and **Standard Dense Vector Search**.
* **System Diagnostics Card:** Displays real-time status of the Pinecone Index, Active LLM Model, Search Mode, **Redis Cache (`Connected ⚡` / `Offline`)**, and Streaming state.
* **Clear Chat Button:** Resets the conversation session state with one click.

---

## ⚡ Redis Exact & Semantic Caching

The project includes a production-grade two-layer caching engine (`src/cache.py`) backed by Redis:

1. **Layer 1 — Exact Match Cache (`<1ms`):** Uses SHA-256 hashes of normalized queries to serve identical repeated questions directly from Redis.
2. **Layer 2 — Semantic Match Cache (`~5-10ms`):** Embeds incoming queries locally with **FastEmbed** (`BAAI/bge-small-en-v1.5`) and checks Cosine Similarity against cached query vectors in Redis (`threshold >= 0.88`). Differently worded questions with the same meaning are served from cache without calling Pinecone or Groq!
3. **Auto Invalidation & TTL:** Cache entries automatically expire after 1 hour (`ttl_seconds=3600`), and `rag_cache.clear()` invalidates stale entries whenever a new document is uploaded.
4. **Live Terminal Logging:** Prints `[CACHE HIT - EXACT]`, `[CACHE HIT - SEMANTIC]`, `[CACHE MISS]`, `[CACHE SET]`, and `[CACHE CLEAR]` in real time to your terminal.
5. **Fail-Open Resilience:** If Redis is offline, the system gracefully falls back to the standard RAG pipeline without crashing.

---

## 🚦 Running the Application

### Option A: Run the Streamlit Web Application (Frontend + Backend)
```bash
uv run streamlit run app_ui.py
```

### Option B: Run the FastAPI REST Server
Start the local FastAPI development server:

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

### 4. Advanced Question (Hybrid Search + Reranking)
* **Endpoint:** `POST /ask-advanced`
* **Features:** Combines Semantic Vector Search (Dense) + Keyword BM25 Search (Sparse), merges candidate lists via **Reciprocal Rank Fusion (RRF)**, and reranks candidates with **`bge-reranker-v2-m3`** before sending to Groq.
* **Request Body:**
  ```json
  {
    "query": "What is the flagship product of NovaTech and what drone technology does it use?"
  }
  ```
* **Response:**
  ```json
  {
    "question": "What is the flagship product of NovaTech and what drone technology does it use?",
    "answer": "NovaTech's flagship product is 'AuraScan 3.0', which uses drone imagery to detect faults in solar power plants.",
    "sources": [
      "NovaTech builds advanced AI systems, and their flagship product is called 'AuraScan 3.0', which detects faults in solar power plants using drone imagery."
    ],
    "pipeline_type": "hybrid_rrf_reranking"
  }
  ```
* **cURL Example:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/ask-advanced" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is AuraScan 3.0?"}'
  ```

---

### 5. Upload File (PDF / TXT Document)
* **Endpoint:** `POST /upload-file`
* **Features:** Automatically extracts text from uploaded PDF or TXT files, chunks them, and stores them in Pinecone.
* **cURL Example:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/upload-file" \
       -F "file=@/path/to/handbook.pdf"
  ```

---

### 6. Conversational Streaming Chat (Token Streaming & Memory)
* **Endpoint:** `POST /chat-stream`
* **Features:** Takes conversational history + current follow-up question, rewrites question if needed, performs Hybrid Search + Reranking, and streams tokens in real time.
* **Request Body:**
  ```json
  {
    "query": "Can I take it in December?",
    "history": [
      {"role": "user", "content": "What is NovaTech's leave policy?"},
      {"role": "assistant", "content": "Employees enjoy 20 days of annual leave and 10 days of sick leave."}
    ],
    "use_hybrid": true
  }
  ```
* **cURL Example:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/chat-stream" \
       -H "Content-Type: application/json" \
       -d '{
         "query": "Can I take it in December?",
         "history": [{"role": "user", "content": "What is the annual leave?"}],
         "use_hybrid": true
       }'
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
