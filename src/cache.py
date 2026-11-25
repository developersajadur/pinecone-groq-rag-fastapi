import hashlib
import json
import os
import numpy as np
from fastembed import TextEmbedding

try:
    import redis
except ImportError:
    redis = None  # Gracefully handled if redis package is not yet installed


class RedisSemanticCache:
    """
    Production-ready two-layer cache for RAG queries backed by Redis:
    1. Exact Match Cache: O(1) lookup for identical queries using SHA-256 keys.
    2. Semantic Cache: Stores query embeddings + answers in Redis and computes
       Cosine Similarity locally using FastEmbed + NumPy.

    Includes automatic Time-To-Live (TTL) expiration via Redis `SETEX` and
    fails open if Redis is temporarily unavailable.
    """

    def __init__(self, similarity_threshold: float = 0.88, ttl_seconds: int = 3600):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._embedder: TextEmbedding | None = None
        self.client = None

        if redis is not None:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6391/0").strip()
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )

    def _get_embedder(self) -> TextEmbedding:
        """Lazy-load FastEmbed model so app startup stays fast."""
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _embed_query(self, text: str) -> list[float]:
        """Convert query text into a normalized 384-dim vector locally."""
        embedder = self._get_embedder()
        vector = list(embedder.embed([text]))[0]
        vec_np = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec_np)
        normalized = vec_np / norm if norm > 0 else vec_np
        return normalized.tolist()

    def _exact_key(self, normalized_query: str) -> str:
        q_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        return f"rag:exact:{q_hash}"

    def _semantic_key(self, normalized_query: str) -> str:
        q_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        return f"rag:semantic:{q_hash}"

    def is_connected(self) -> bool:
        """Ping Redis to check live connection status."""
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def get(self, query: str) -> dict | None:
        """
        Look up a query in Exact Cache first, then Semantic Cache.
        Returns cached result dict on HIT, or None on MISS.
        """
        if self.client is None:
            return None

        normalized = query.strip().lower()

        try:
            # 1. Exact Match Check (O(1) in Redis)
            exact_raw = self.client.get(self._exact_key(normalized))
            if exact_raw:
                data = json.loads(exact_raw)
                print(
                    f"\n⚡ [CACHE HIT - EXACT] Query: '{query}' | Similarity: 1.0000",
                    flush=True,
                )
                return {
                    "answer": data["answer"],
                    "sources": data["sources"],
                    "cache_type": "redis_exact",
                    "similarity": 1.0,
                }

            # 2. Semantic Match Check across active Redis semantic keys
            semantic_keys = list(self.client.scan_iter(match="rag:semantic:*"))
            if not semantic_keys:
                print(f"\n🔍 [CACHE MISS] Query: '{query}' (No cached entries yet)", flush=True)
                return None

            query_vec = np.array(self._embed_query(normalized), dtype=np.float32)
            best_score = -1.0
            best_data = None

            # Fetch all semantic entries in one network round-trip using MGET
            raw_entries = self.client.mget(semantic_keys)
            for raw in raw_entries:
                if not raw:
                    continue
                entry = json.loads(raw)
                entry_vec = np.array(entry["vector"], dtype=np.float32)
                score = float(np.dot(query_vec, entry_vec))
                if score > best_score:
                    best_score = score
                    best_data = entry

            if best_data and best_score >= self.similarity_threshold:
                print(
                    f"\n⚡ [CACHE HIT - SEMANTIC] Query: '{query}' | Matched: '{best_data['query']}' | Similarity: {round(best_score, 4)}",
                    flush=True,
                )
                return {
                    "answer": best_data["answer"],
                    "sources": best_data["sources"],
                    "cache_type": "redis_semantic",
                    "similarity": round(best_score, 4),
                }

            print(
                f"\n🔍 [CACHE MISS] Query: '{query}' (Best Similarity: {round(best_score, 4)} < {self.similarity_threshold})",
                flush=True,
            )

        except Exception as e:
            # Fail open: if Redis is down, continue with normal RAG pipeline
            print(f"\n⚠️ [CACHE ERROR] Redis lookup failed: {e}", flush=True)
            return None

        return None

    def set(self, query: str, answer: str, sources: list[str]) -> None:
        """Store a new query-answer pair in both Exact and Semantic Redis caches."""
        if self.client is None:
            return

        normalized = query.strip().lower()
        try:
            # Store Exact Match with automatic Redis TTL
            exact_payload = json.dumps({"answer": answer, "sources": sources})
            self.client.setex(self._exact_key(normalized), self.ttl_seconds, exact_payload)

            # Store Semantic Entry with automatic Redis TTL
            vec = self._embed_query(normalized)
            semantic_payload = json.dumps({
                "query": normalized,
                "vector": vec,
                "answer": answer,
                "sources": sources,
            })
            self.client.setex(self._semantic_key(normalized), self.ttl_seconds, semantic_payload)
            print(f"💾 [CACHE SET] Stored in Redis for: '{query}' (TTL: {self.ttl_seconds}s)", flush=True)

        except Exception as e:
            print(f"⚠️ [CACHE SET ERROR] {e}", flush=True)

    def clear(self) -> None:
        """Invalidate all RAG cache keys in Redis when new documents are indexed."""
        if self.client is None:
            return
        try:
            keys = list(self.client.scan_iter(match="rag:*"))
            if keys:
                self.client.delete(*keys)
                print(f"🧹 [CACHE CLEAR] Cleared {len(keys)} stale cache key(s) from Redis.", flush=True)
        except Exception as e:
            print(f"⚠️ [CACHE CLEAR ERROR] {e}", flush=True)


# Global singleton instance
rag_cache = RedisSemanticCache(similarity_threshold=0.88, ttl_seconds=3600)
