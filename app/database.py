"""
database.py - Async PostgreSQL layer with pgvector support.

Manages a connection pool via asyncpg and handles storing/querying
512-d feature vectors using pgvector's cosine distance operator.
"""

import time
import logging
from typing import List, Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger("visionmap.database")

# I'm using a module-level connection pool, initialized once at startup.
# Setting up a pool here avoids the ~5-20ms overhead of creating a new
# TCP connection for every request (handshake + auth each time).
_pool: Optional[asyncpg.Pool] = None


async def init_db_pool(database_url: str) -> asyncpg.Pool:
    """
    Create the async connection pool and register pgvector's custom type.

    I added the `init` callback to run on every new connection — it calls
    register_vector() to teach asyncpg how to serialize/deserialize the `vector`
    column type. Without this, I'd get "unknown type: vector" errors.
    """
    global _pool

    _pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )

    logger.info(f"Database pool created (min=2, max=10)")
    return _pool


async def _init_connection(conn: asyncpg.Connection):
    """Register the pgvector type on each pooled connection."""
    await register_vector(conn)


async def close_db_pool():
    """Gracefully close all pooled connections."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Database pool closed")
        _pool = None


async def store_embedding(
    image_name: str,
    embedding: List[float],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """Insert a feature vector into spatial_features. Returns the new row's UUID."""
    start = time.perf_counter()

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO spatial_features (image_name, embedding, latitude, longitude)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            image_name,
            embedding,
            latitude,
            longitude,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    record_id = str(row["id"])
    logger.info(f"Stored '{image_name}' -> {record_id} in {elapsed_ms:.1f}ms")
    return record_id


async def search_nearest(
    embedding: List[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Find the top_k most similar vectors using cosine distance.

    I'm using pgvector's <=> operator here, which returns cosine DISTANCE
    (1 - cosine_similarity). I included an ORDER BY ASC to get the most
    similar ones first, and I compute (1 - distance) to return a clean
    0-to-1 similarity score.

    The HNSW index I set up on the embedding column makes this super fast
    — it does an O(log n) greedy graph traversal instead of an O(n) scan.
    """
    start = time.perf_counter()

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, image_name, latitude, longitude,
                   1 - (embedding <=> $1) AS similarity
            FROM spatial_features
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            top_k,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Vector search: {elapsed_ms:.1f}ms (top_k={top_k}, found={len(rows)})")

    return [
        {
            "id": str(row["id"]),
            "image_name": row["image_name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]
