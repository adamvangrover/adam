from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.api.routers import agents
from contextlib import asynccontextmanager
import logging
import traceback

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
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)},
    )

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])


@app.get("/")
async def root():
    return {"message": "Welcome to Adam Core API. Use /docs for Swagger UI."}


def start():
    import uvicorn
    # Use standard port 8000
    uvicorn.run("core.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
