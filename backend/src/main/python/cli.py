import click
import json # For pretty printing dicts/lists
import os # For debugging
import sys # For debugging

print(f"DEBUG cli.py: Initial os.getcwd(): {os.getcwd()}")
print(f"DEBUG cli.py: Initial sys.path: {sys.path}")

# To make this runnable from project root, ensure PYTHONPATH includes backend/src/main/python
# e.g. by setting PYTHONPATH=. when in project root, or by installing the backend as a package.
# For simplicity here, we assume the imports will work based on typical project structure.
try:
    from .reasoning_engine import ReasoningEngine
    from .kg_builder import get_kg_instance, build_graph_from_csvs, KnowledgeGraph # For init-data
except ImportError:
    # Fallback for cases where the script might be run directly and '.' imports fail
    # This is common in simpler Click script setups if not installed as a package.
    # You might need to adjust your PYTHONPATH environment variable:
    # export PYTHONPATH=$PYTHONPATH:/path/to/your/project
    # Or run as `python -m backend.src.main.python.cli ...` from project root.
    print("ImportError: Could not import local modules. Ensure PYTHONPATH is set correctly or run as a module.")
    print("Attempting to import assuming backend.src.main.python is in PYTHONPATH")
    from reasoning_engine import ReasoningEngine
    from kg_builder import get_kg_instance, build_graph_from_csvs, KnowledgeGraph


# Global engine instance, initialized on first command that needs it
# This leverages the lazy loading in get_kg_instance()
ENGINE_INSTANCE = None

def get_engine():
    global ENGINE_INSTANCE
    if ENGINE_INSTANCE is None:
        ENGINE_INSTANCE = ReasoningEngine()
    return ENGINE_INSTANCE

@click.group()
def cli():
    """
    Narrative Library CLI to interact with company data, drivers, and explanations.
    """
    pass

# --- Report Commands ---
@cli.group()
def report():
    """Commands for generating reports."""
    pass

@report.command(name="driver-impacts")
@click.option(
    '--output-file',
    default="high_impact_driver_report.jsonl",
    help="Path to the output JSONL file.",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True
)
@click.option(
    '--min-probability',
    default=0.65,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum probability for an impact to be included.",
    show_default=True
)
def report_driver_impacts(output_file: str, min_probability: float):
    """Generates a JSONL report of high-probability driver impacts."""
    click.echo(f"Generating driver impact report with min probability {min_probability} to file {output_file}...")

    # Debugging imports:
    import os
    import sys
    click.echo(f"DEBUG: Current sys.path: {sys.path}")
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    click.echo(f"DEBUG: cli.py directory: {cli_dir}")
    tools_path = os.path.join(cli_dir, "tools")
    click.echo(f"DEBUG: Calculated tools path: {tools_path}")
    click.echo(f"DEBUG: Does tools path exist? {os.path.exists(tools_path)}")
    tools_init_path = os.path.join(tools_path, "__init__.py")
    click.echo(f"DEBUG: Does tools/__init__.py exist? {os.path.exists(tools_init_path)}")
    report_script_path = os.path.join(tools_path, "generate_driver_impact_report.py")
    click.echo(f"DEBUG: Does tools/generate_driver_impact_report.py exist? {os.path.exists(report_script_path)}")
    # End Debugging

    try:
        # Explicit full path import
        from backend.src.main.python.tools.generate_driver_impact_report import generate_report as generate_driver_impact_report_func
    except ImportError as e:
        click.secho(f"Error: Could not import the report generator: {e}", fg="red")
        click.secho("Ensure PYTHONPATH includes the project root (e.g., `export PYTHONPATH=.`) OR this script is run as a module.", fg="red")
        return

    try:
        generate_driver_impact_report_func(output_filepath=output_file, min_probability=min_probability)
        click.secho(f"Report generation process finished. See output at: {output_file}", fg="green")
    except Exception as e:
        click.secho(f"An error occurred during report generation: {e}", fg="red")

# --- Data Initialization Commands ---
@cli.command(name="init-data")
@click.option('--force', is_flag=True, help="Force re-initialization of data even if already loaded.")
def init_data(force: bool):
    """
    Initializes or re-initializes the Knowledge Graph from CSV data.
    The KG normally loads lazily, but this command allows explicit control.
    """
    click.echo("Initializing Knowledge Graph data...")
    try:
        # To truly re-initialize, we might need to replace the global KG_INSTANCE in kg_builder
        # For now, get_kg_instance handles lazy loading. If 'force' is used, we can try to rebuild.
        # This is a simplified re-init for CLI demonstration.
        current_kg = get_kg_instance() # Ensures it's loaded at least once
        if force:
            click.echo("Forcing data reload...")
            # This is a conceptual reload; actual replacement of the singleton KG_INSTANCE
            # used by the engine might need more direct manipulation if not passed around.
            # For this CLI, creating a new KG and engine for this command's scope is okay.
            new_kg = KnowledgeGraph()
            build_graph_from_csvs(new_kg)
            global ENGINE_INSTANCE # Modify the CLI's global engine
            ENGINE_INSTANCE = ReasoningEngine(kg=new_kg)
            click.echo(f"Data reloaded. New KG has {len(new_kg.nodes)} nodes, {len(new_kg.edges)} edges.")
        else:
            click.echo(f"Knowledge Graph already loaded/initialized. Nodes: {len(current_kg.nodes)}, Edges: {len(current_kg.edges)}.")
            click.echo("Use --force to reload data.")

    except Exception as e:
        click.secho(f"Error initializing data: {e}", fg="red")

@cli.group()
def company():
    """Commands for company-specific information."""
    pass

@company.command(name="drivers")
@click.argument('company_id')
def company_drivers(company_id: str):
    """Display all drivers for a given COMPANY_ID."""
    engine = get_engine()
    company_id_upper = company_id.upper()
    company_node = engine.kg.get_node(company_id_upper)

    if not company_node or company_node.label != "Company":
        click.secho(f"Company '{company_id_upper}' not found.", fg="red")
        return

    click.echo(f"Drivers for {company_node.properties.get('name', company_id_upper)} ({company_id_upper}):")
    drivers_data = engine.get_all_company_drivers(company_id_upper)
    if not drivers_data:
        click.echo("No drivers found for this company.")
        return

    for i, driver in enumerate(drivers_data):
        click.echo(f"\nDriver {i+1}: {driver.get('name', 'N/A')} ({driver.get('id', 'N/A')})")
        click.echo(f"  Type: {driver.get('type', 'N/A')}")
        click.echo(f"  Description: {driver.get('description', 'N/A')}")
        click.echo(f"  Impact: {driver.get('impactPotential', 'N/A')}, Horizon: {driver.get('timeHorizon', 'N/A')}")
        if driver.get('metrics'):
            click.echo(f"  Metrics: {json.dumps(driver.get('metrics'))}")


@company.command(name="explain")
@click.argument('company_id')
@click.option('--template-id', default="TPL_COMPANY_OVERVIEW_STANDARD", help="ID of the narrative template to use.", show_default=True)
def company_explain(company_id: str, template_id: str):
    """Display LLM-generated explanation for a COMPANY_ID using a specific template."""
    engine = get_engine()
    company_id_upper = company_id.upper()
    company_node = engine.kg.get_node(company_id_upper)

    if not company_node or company_node.label != "Company":
        click.secho(f"Company '{company_id_upper}' not found.", fg="red")
        return

    click.echo(f"Generating explanation for {company_node.properties.get('name', company_id_upper)} ({company_id_upper}) using template '{template_id}'...")
    explanation = engine.generate_narrative_explanation_with_llm(company_id_upper, template_id=template_id)

    if explanation.get("error"):
        click.secho(f"Error generating explanation: {explanation['error']}", fg="red")
        return

    click.echo("\n--- Narrative Summary ---")
    click.echo(explanation.get("narrative_summary", "No narrative summary available."))

    click.echo(f"\n--- Details ---")
    click.echo(f"Company Name: {explanation.get('company_name', 'N/A')}")
    click.echo(f"Number of Drivers Found: {explanation.get('num_drivers_found', 'N/A')}")

    if explanation.get("drivers"):
        click.echo("\nAssociated Drivers:")
        for driver in explanation.get("drivers", []):
            click.echo(f"  - {driver.get('name', 'N/A')} ({driver.get('id', 'N/A')}) - Type: {driver.get('type', 'N/A')}")


@cli.group()
def driver():
    """Commands for driver-specific information."""
    pass

@driver.command(name="details")
@click.argument('driver_id')
def driver_details(driver_id: str):
    """Display details for a specific DRIVER_ID."""
    engine = get_engine()
    driver_id_upper = driver_id.upper()
    details = engine.get_driver_details(driver_id_upper)

    if not details:
        click.secho(f"Driver '{driver_id_upper}' not found.", fg="red")
        return

    click.echo(f"Details for Driver: {details.get('name', 'N/A')} ({driver_id_upper})")
    click.echo(f"  ID: {details.get('id', 'N/A')}")
    click.echo(f"  Type: {details.get('type', 'N/A')}")
    click.echo(f"  Description: {details.get('description', 'N/A')}")
    click.echo(f"  Impact Potential: {details.get('impactPotential', 'N/A')}")
    click.echo(f"  Time Horizon: {details.get('timeHorizon', 'N/A')}")
    if details.get('metrics'):
        click.echo(f"  Metrics: {json.dumps(details.get('metrics'))}")
    if details.get('relatedMacroFactorIds'):
        click.echo(f"  Related Macro Factors: {', '.join(details.get('relatedMacroFactorIds', []))}")

if __name__ == '__main__':
    # This makes the script executable.
    # To use:
    # python backend/src/main/python/cli.py company explain AAPL
    # python backend/src/main/python/cli.py driver details DRV001
    #
    # For more idiomatic Click usage, you'd set up an entry point in pyproject.toml or setup.py
    # (e.g., narrative-cli = backend.src.main.python.cli:cli)
    # and then just run `narrative-cli company explain AAPL`
    cli()

# Temporarily make driver-impacts a direct command for testing -m execution
# @cli.group()
# def report():
#     """Commands for generating reports."""
#     pass

@cli.command(name="driver-impacts") # Changed from @report.command
@click.option(
    '--output-file',
    default="high_impact_driver_report.jsonl",
    help="Path to the output JSONL file.",
    type=click.Path(dir_okay=False, writable=True), # Ensures file path is writable
    show_default=True
)
@click.option(
    '--min-probability',
    default=0.65,
    type=click.FloatRange(0.0, 1.0), # Restrict to valid probability range
    help="Minimum probability for an impact to be included.",
    show_default=True
)
def report_driver_impacts(output_file: str, min_probability: float):
    """Generates a JSONL report of high-probability driver impacts."""
    click.echo(f"Generating driver impact report with min probability {min_probability} to file {output_file}...")

    # Ensure the tools module and its script can be imported
    # This might require specific PYTHONPATH setup if not run as an installed package.
    try:
        from .tools import generate_driver_impact_report as report_generator
    except ImportError:
        click.secho("Error: Could not import the report generator. Ensure PYTHONPATH is correctly set up.", fg="red")
        click.secho("Try running with `PYTHONPATH=. python backend/src/main/python/cli.py ...` from project root.", fg="red")
        return

    try:
        # The generate_report function in the script handles its own output and print statements.
        report_generator.generate_report(output_filepath=output_file, min_probability=min_probability)
        click.secho(f"Report generation process finished. See output at: {output_file}", fg="green")
    except Exception as e:
        click.secho(f"An error occurred during report generation: {e}", fg="red")
