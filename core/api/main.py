from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.api.routers import agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize things if needed
    print("Adam Core API starting up...")
    yield
    # Shutdown
    print("Adam Core API shutting down...")

app = FastAPI(
    title="Adam Core API",
    description="Microservice Gateway for Adam Autonomous Financial Analyst",
    version="23.5.0",
    lifespan=lifespan
)

app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])

@app.get("/")
async def root():
    return {"message": "Welcome to Adam Core API. Use /docs for Swagger UI."}

def start():
    import uvicorn
    uvicorn.run("core.api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    start()
