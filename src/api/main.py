from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ingest

app = FastAPI(
    title="Data Ingestion Engine & LLM Wrapper",
    description="A portable, modular system for parsing unstructured/semi-structured data.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingestion"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
