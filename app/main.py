"""
main.py - FastAPI application for the VisionMap VPS pipeline.

Two main endpoints:
  POST /ingest    -> Upload an image, extract features, store in Postgres
  POST /localize  -> Upload an image, find the most similar stored image
"""

import os
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app.engine import FeatureExtractor
from app.database import init_db_pool, close_db_pool, store_embedding, search_nearest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visionmap.api")

# Global reference to my ML model — I load this once at startup and reuse it for all requests
extractor: Optional[FeatureExtractor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup/shutdown.

    I structured this to load the ML model and create the DB connection pool
    before the app starts serving, and then clean everything up on shutdown.
    I chose to use this pattern instead of the older @app.on_event("startup").
    """
    global extractor

    logger.info("Starting VisionMap API...")

    # Load the model once (takes ~1-2s) instead of per-request
    extractor = FeatureExtractor()

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://visionmap:visionmap@localhost:5433/visionmap",
    )
    await init_db_pool(database_url)

    logger.info("VisionMap API ready")
    yield

    logger.info("Shutting down...")
    await close_db_pool()
    extractor = None


app = FastAPI(
    title="VisionMap",
    description="Visual Positioning System — image ingestion and spatial localization via CNN embeddings + pgvector.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    image_name: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """
    Upload an image, extract its 512-d feature embedding, and store it
    in PostgreSQL for later similarity search.
    """
    total_start = time.perf_counter()

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # KEY DESIGN DECISION: I offload PyTorch inference to a background thread.
    # Because FastAPI runs on an async event loop, and PyTorch's forward pass is
    # synchronous and CPU/GPU-bound, calling it directly would block the loop and
    # freeze ALL other requests for the duration of inference (~50-200ms).
    # I use asyncio.to_thread() to keep the event loop free to handle concurrent traffic.
    inference_start = time.perf_counter()
    embedding = await asyncio.to_thread(extractor.extract, image_bytes)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # DB insert is already async via asyncpg — no blocking here
    db_start = time.perf_counter()
    record_id = await store_embedding(image_name, embedding, latitude, longitude)
    db_ms = (time.perf_counter() - db_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    logger.info(
        f"/ingest '{image_name}' -> {record_id} | "
        f"inference={inference_ms:.1f}ms, db={db_ms:.1f}ms, total={total_ms:.1f}ms"
    )

    return {
        "status": "ok",
        "id": record_id,
        "image_name": image_name,
        "embedding_dims": len(embedding),
        "timing": {
            "inference_ms": round(inference_ms, 1),
            "db_insert_ms": round(db_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


@app.post("/localize")
async def localize(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Upload a query image and find the most similar images in the database.
    I built this to simulate how a VPS figures out where a user is standing —
    it matches the uploaded photo against my database of geolocated reference images.
    """
    total_start = time.perf_counter()

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Same thread-offloading pattern as /ingest
    inference_start = time.perf_counter()
    embedding = await asyncio.to_thread(extractor.extract, image_bytes)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # Cosine similarity search, accelerated by the HNSW index
    db_start = time.perf_counter()
    matches = await search_nearest(embedding, top_k=top_k)
    db_ms = (time.perf_counter() - db_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="No images in database yet. Ingest some images first.",
        )

    logger.info(
        f"/localize -> '{matches[0]['image_name']}' "
        f"(sim={matches[0]['similarity']:.4f}) | "
        f"inference={inference_ms:.1f}ms, db={db_ms:.1f}ms, total={total_ms:.1f}ms"
    )

    return {
        "status": "ok",
        "matches": matches,
        "timing": {
            "inference_ms": round(inference_ms, 1),
            "db_search_ms": round(db_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


@app.get("/health")
async def health():
    """Simple readiness check."""
    return {
        "status": "healthy",
        "model_loaded": extractor is not None,
        "device": str(extractor.device) if extractor else None,
    }
