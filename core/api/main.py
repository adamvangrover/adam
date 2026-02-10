from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.api.routers import agents
from core.api.deps import verify_api_key
from core.settings import settings
from contextlib import asynccontextmanager
import logging
import traceback
import os

# Setup logger
logger = logging.getLogger("core.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize things if needed
    logger.info("Adam Core API starting up...")
    yield
    # Shutdown
    logger.info("Adam Core API shutting down...")

app = FastAPI(
    title="Adam Core API",
    description="Microservice Gateway for Adam Autonomous Financial Analyst",
    version="23.5.0",
    lifespan=lifespan
)

# Robust Error Handling Middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {str(exc)}")
    logger.error(traceback.format_exc())

    # 🛡️ Sentinel: Don't leak internal details in production
    if settings.debug:
        content = {"message": "Internal Server Error", "details": str(exc)}
    else:
        content = {"message": "Internal Server Error"}

    return JSONResponse(
        status_code=500,
        content=content,
    )

# 🛡️ Sentinel: Restrict CORS to specific origins
# Allow configuring via environment variable, default to common development ports
allowed_origins_str = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000"
)
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🛡️ Sentinel: Enforce API Key Authentication on Agents API
app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["agents"],
    dependencies=[Depends(verify_api_key)]
)


@app.get("/")
async def root():
    return {"message": "Welcome to Adam Core API. Use /docs for Swagger UI."}


def start():
    import uvicorn
    # Use standard port 8000
    uvicorn.run("core.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
