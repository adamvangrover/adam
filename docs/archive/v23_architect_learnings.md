# Apex Architect: Learnings & Patterns

## 1. The "Additive" Principle in Practice
When upgrading `GenerativeRiskEngine`, we demonstrated that subclassing (`StochasticRiskEngine(GenerativeRiskEngine)`) allows for radical capability expansion (Merton Jump-Diffusion, Cholesky Decomposition) without touching the original logic.

## 2. Backward Compatibility via Pydantic
By using `Optional[str] = Field(None, ...)` for new fields in `MarketScenario`, we ensured that legacy code paths (which don't supply these fields) continue to function without modification.

## 3. Financial Modeling
We implemented the **Merton Jump-Diffusion Model (1976)** to better capture the "fat tails" observed in modern credit markets, moving beyond the limitation of simple Gaussian assumptions.
