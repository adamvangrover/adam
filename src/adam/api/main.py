from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
import torch
from src.adam.api.models import OptimizationRequest, OptimizationResponse
from src.adam.api.auth import get_current_user
from src.adam.core.optimizers import AdamW, Lion, AdamMini
from src.adam.core.state_manager import StateManager

# Initialize State Manager (tries Redis, falls back to memory)
state_manager = StateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources
    yield
    # Clean up resources

app = FastAPI(
    title="Adam Optimization Service",
    description="Microservice for State-of-the-Art Optimization Algorithms (Stateful)",
    version="1.1.0",
    lifespan=lifespan
)

@app.post("/optimize", response_model=OptimizationResponse, dependencies=[Depends(get_current_user)])
async def optimize(request: OptimizationRequest):
    """
    Apply a single optimization step with state persistence.
    """
    try:
        # Convert to tensor (CPU)
        params = torch.tensor(request.parameters, dtype=torch.float32)
        grads = torch.tensor(request.gradients, dtype=torch.float32)

        # Attach gradients
        params.grad = grads

        algo = request.config.algorithm.lower()

        # Initialize Optimizer
        # We must re-initialize the optimizer object because this is a stateless HTTP request context,
        # but we will hydrate it with state from Redis.
        if algo == "adamw":
            opt = AdamW([params],
                        lr=request.config.learning_rate,
                        betas=tuple(request.config.betas),
                        weight_decay=request.config.weight_decay,
                        eps=request.config.epsilon)
        elif algo == "lion":
            opt = Lion([params],
                       lr=request.config.learning_rate,
                       betas=tuple(request.config.betas),
                       weight_decay=request.config.weight_decay)
        elif algo == "adam-mini":
            opt = AdamMini([params],
                           lr=request.config.learning_rate,
                           betas=tuple(request.config.betas),
                           weight_decay=request.config.weight_decay,
                           eps=request.config.epsilon)
        else:
             raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo}")

        # Load State
        state_key = f"opt_state:{request.session_id}"
        saved_state = state_manager.load_state(state_key)

        if saved_state:
            try:
                # PyTorch's load_state_dict handles mapping state to the new param tensors
                # provided the structure (1 group, 1 param) is identical.
                opt.load_state_dict(saved_state)
            except Exception as e:
                # If state is incompatible (e.g. different algo or param size), we might want to reset or error.
                # For now, we log and proceed (which resets state).
                print(f"Warning: Failed to load state: {e}. Resetting state.")

        # Perform Step
        opt.step()

        # Save State
        # We serialize the state_dict (which contains tensors)
        new_state = opt.state_dict()
        state_manager.save_state(state_key, new_state)

        return OptimizationResponse(
            updated_parameters=params.detach().tolist(),
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "persistence": "redis" if state_manager.using_redis else "memory"}
