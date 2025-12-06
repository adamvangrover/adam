from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """
    Load requirements from a pip requirements file.
    
    This function is designed to be robust and flexible:
    - Checks for file existence.
    - Filters out comments and empty lines.
    - Handles flags (like --index-url) by stripping them, ensuring only 
      valid package identifiers are passed to setuptools.
    """
    requirements = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                # Use a generator for efficient line reading
                lineiter = (line.strip() for line in f)
                for line in lineiter:
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    
                    # Skip pure flag lines (e.g. "--extra-index-url ...")
                    if line.startswith("-"):
                        continue

                    # Split by space to handle lines like 'package==1.0 --option'
                    # taking the first part as the requirement
                    parts = line.split()
                    if parts:
                        req = parts[0]
                        # Ensure the requirement itself isn't a flag
                        if not req.startswith("-"):
                            requirements.append(req)
        except Exception as e:
            print(f"Warning: Error parsing {filename}: {e}")
            # Return empty to trigger fallback or empty install
            return []
            
    return requirements

# Dynamic Requirement Loading
# Try to load from requirements.txt, but maintain a fallback list for resilience.
try:
    install_requires = parse_requirements('requirements.txt')
except Exception:
    install_requires = []

# Fallback: Core dependencies required for basic functionality.
# This ensures the package is installable even if requirements.txt is missing 
# or during isolated builds.
if not install_requires:
    print("Info: Requirements file not found or empty. Using fallback core dependencies.")
    install_requires = [
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ]

setup(
    name='adam-project', # Using the more expansive name to cover the full platform
    version='23.5.0',
    description='Adam: Autonomous Financial Analyst Agent Platform',
    author='Adam Team',
    
    # Expansive Package Discovery:
    # Includes 'core' logic as well as 'services' (e.g., webapp, specialized microservices)
    # to support a monorepo or unified build structure.
    packages=find_packages(include=['core', 'core.*', 'services', 'services.*']),
    
    include_package_data=True,
    install_requires=install_requires,
    python_requires='>=3.9',
    
    # Flexible Entry Points:
    # Provides aliases for the general runner, the specific API server, 
    # and a standard 'adam' command for ease of use.
    entry_points={
        'console_scripts': [
            'adam=core.main:main',           # Unified CLI entry
            'adam-run=core.main:main',       # Explicit runner alias
            'adam-api=core.api.main:start',  # Direct API server entry
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Future-Proofing:
    # Extras definitions allow for leaner installs or developer setups
    # without cluttering the main dependency list.
    extras_require={
        'dev': ['pytest', 'black', 'flake8', 'mypy'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'ui': ['streamlit'] # Placeholder for UI-specific libs
    }
)