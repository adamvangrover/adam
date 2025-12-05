from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                # Split off any flags like --index-url
                req = line.split(' ')[0]
                requirements.append(req)
    return requirements

# Attempt to read requirements.txt, fallback to minimal if not found or issues
try:
    reqs = parse_requirements('requirements.txt')
except Exception:
    reqs = [
        "pandas",
        "numpy",
        "pydantic",
        "requests",
        "fastapi",
        "uvicorn"
    ]

setup(
    name="adam-core",
    version="23.5.0",
    description="Adam: Autonomous Financial Analyst - Core System",
    author="Adam Team",
    packages=find_packages(include=['core', 'core.*']),
    install_requires=reqs,
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'adam-run=core.main:main',
            'adam-api=core.api.main:start',
        ],
    },
)
