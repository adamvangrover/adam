from fastapi import FastAPI
from adam_os.api.routes import ledger

app = FastAPI(
    title="ADAM OS API",
    description="Mission-critical, event-driven financial operating system",
    version="0.1.0",
)

app.include_router(ledger.router)

@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy"}
