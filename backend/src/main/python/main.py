from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Optional

# Use relative imports for local modules
from .reasoning_engine import ReasoningEngine
from .kg_builder import get_kg_instance, build_graph_from_csvs # For a potential reload endpoint
from .knowledge_graph import KnowledgeGraph
# For response models, we can use Pydantic models that mirror our data classes
# from .models import Company, Driver # Pydantic versions of these

# Initialize FastAPI app
app = FastAPI(
    title="Narrative Library API",
    description="API for accessing financial narrative and driver information.",
    version="0.1.0"
)

# Global engine instance
# The KG is built when get_kg_instance() is first called by ReasoningEngine
engine: Optional[ReasoningEngine] = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("Application startup: Initializing Reasoning Engine...")
    # This will trigger kg_builder.get_kg_instance() which builds the graph if not already built
    engine = ReasoningEngine()
    print("Reasoning Engine initialized.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Narrative Library API. KG is loaded."}

@app.post("/admin/reload-data", summary="Reloads data from CSVs into the Knowledge Graph")
async def reload_data():
    global engine
    try:
        print("Admin: Reloading data into Knowledge Graph...")
        # Create a new KG instance and rebuild
        new_kg = KnowledgeGraph()
        build_graph_from_csvs(new_kg) # This function populates the passed KG object

        # Replace the engine's KG instance or create a new engine
        engine = ReasoningEngine(kg=new_kg)
        # Or if KG_INSTANCE in kg_builder is used by ReasoningEngine directly:
        # from .kg_builder import KG_INSTANCE
        # KG_INSTANCE = new_kg
        # engine = ReasoningEngine() # it will pick up the new KG_INSTANCE

        print("Data reloaded successfully.")
        return {"message": "Knowledge Graph data reloaded successfully."}
    except Exception as e:
        print(f"Error reloading data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")


@app.get("/companies/{company_id}/drivers", summary="Get all drivers for a company")
async def get_company_drivers(company_id: str) -> List[Dict]:
    if not engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")

    drivers = engine.get_all_company_drivers(company_id.upper()) # Assuming IDs are uppercase like AAPL
    if not drivers and not engine.kg.get_node(company_id.upper()):
         raise HTTPException(status_code=404, detail=f"Company with ID '{company_id}' not found.")
    return drivers

@app.get("/companies/{company_id}/explanation", summary="Get an explanation for a company's drivers using a specific template")
async def get_company_explanation(company_id: str, template_id: Optional[str] = Query("TPL_COMPANY_OVERVIEW_STANDARD", description="The ID of the narrative template to use.")) -> Dict:
    if not engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")

    explanation = engine.generate_narrative_explanation_with_llm(company_id.upper(), template_id=template_id)

    if explanation.get("error"):
        raise HTTPException(status_code=404, detail=f"Could not generate explanation for company ID '{company_id}' using template '{template_id}': {explanation['error']}")
    return explanation

@app.get("/drivers/{driver_id}", summary="Get details for a specific driver")
async def get_driver_info(driver_id: str) -> Dict:
    if not engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")

    driver_details = engine.get_driver_details(driver_id.upper())
    if not driver_details:
        raise HTTPException(status_code=404, detail=f"Driver with ID '{driver_id}' not found.")
    return driver_details

@app.get("/industries", summary="List all industries")
async def list_industries() -> List[Dict]:
    if not engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    industry_nodes = engine.kg.find_nodes_by_label("Industry")
    return [node.properties for node in industry_nodes]

@app.get("/companies", summary="List all companies (IDs and names)")
async def list_companies(limit: int = 20) -> List[Dict]:
    if not engine:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    company_nodes = engine.kg.find_nodes_by_label("Company")
    return [
        {"id": node.id, "name": node.properties.get("name")}
        for node in company_nodes[:limit]
    ]

# To run this app (from the root directory of the project):
# PYTHONPATH=. uvicorn backend.src.main.python.main:app --reload --port 8000
# (Make sure your PWD is the project root so `backend.src.main.python.main` is resolvable,
# or adjust PYTHONPATH accordingly, e.g. `PYTHONPATH=backend/src/main/python uvicorn main:app --reload`)
# For this tool environment, direct execution is not possible.

if __name__ == "__main__":
    # This part is for conceptual testing if you were to run main.py directly,
    # but uvicorn is the standard way.
    print("To run the API server, use Uvicorn:")
    print("PYTHONPATH=. uvicorn backend.src.main.python.main:app --reload --port 8000")
    print("Example command assuming you are in the project's root directory.")
    print("Ensure your CSV files are in data/sample_data/ relative to the project root.")

    # Basic test to ensure KG loading works when this file is run (though not as server)
    # This is more for sanity check during development.
    try:
        print("Attempting to initialize KG for standalone check...")
        get_kg_instance() # This will build the graph
        print("KG initialized for standalone check.")
        engine_test = ReasoningEngine()
        print("Test: Fetching AAPL explanation...")
        aapl_expl = engine_test.generate_basic_explanation("AAPL")
        print(aapl_expl.get("narrative_summary", "Could not get AAPL explanation."))
    except Exception as e:
        print(f"Error during standalone check: {e}")
        print("This might be due to relative path issues when not run via uvicorn from project root.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected DATA_DIR relative to kg_builder.py: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'sample_data')}")

# Note on PYTHONPATH for execution:
# If your project root is 'narrative_library_project', and this file is in
# 'narrative_library_project/backend/src/main/python/main.py',
# you would typically run uvicorn from 'narrative_library_project':
# `uvicorn backend.src.main.python.main:app --reload`
# The `PYTHONPATH=.` (or adding project root to PYTHONPATH) helps Python resolve modules
# like `from .kg_builder import ...` or `from backend.src.main.python.models import ...`
# correctly. The relative imports `from .module` should work if `main.py` is run as part of a package.
